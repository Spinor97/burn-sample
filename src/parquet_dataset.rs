use std::{collections::HashMap, fs::File, path::PathBuf, sync::{Mutex, MutexGuard}};

use arrow::{array::{AsArray, PrimitiveArray}, compute::cast, datatypes::Float32Type};
use burn::data::dataset::Dataset;
use parquet::arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder};
use polars::lazy::{dsl::len, frame::{LazyFrame, ScanArgsParquet}};

use crate::utils::sink_to_local;

#[derive(Clone, Debug)]
pub struct ParquetItem {
    pub feature: Vec<f32>,
    pub label: f32,
}

pub struct Chunk {
    pub features: Vec<Vec<f32>>,
    pub labels: Vec<f32>,
}

pub struct ParquetDataset {
    path: String,
    features: Vec<String>,
    lables: String,
    ttl_rows: usize,
    rg_offset: Vec<usize>,
    projection_indices: Vec<usize>,
    cache: Mutex<HashMap<usize, Chunk>>,
    max_cached: usize,
}

impl ParquetDataset {
    pub fn new(
        path: &str,
        features: Vec<String>,
        labels: String,
        max_cached: usize,
    ) -> Self {
        let file = File::open(path).expect("Fail to open the data file!!!");
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("Fail to read metadata");

        let metadata = builder.metadata().clone();
        let num_row_groups = metadata.num_row_groups();
        let schema_dbsr = metadata.file_metadata().schema_descr();

        let mut rg_offsets = Vec::with_capacity(num_row_groups);
        let mut ttl: usize = 0;

        for i in 0..num_row_groups {
            rg_offsets.push(ttl);
            ttl += metadata.row_group(i).num_rows() as usize;
        }

        let resolve = |name: &str| -> usize {
            (0..schema_dbsr.num_columns())
                .find(|&i| schema_dbsr.column(i).name() == name)
                .unwrap_or_else(|| panic!("Column \"{}\" is not found", name))
        };

        let mut projection_indices: Vec<usize> = features
            .iter()
            .map(|name| resolve(name))
            .collect();

        projection_indices.push(resolve(&labels));

        Self {
            path: path.to_string(),
            features: features,
            lables: labels,
            ttl_rows: ttl,
            rg_offset: rg_offsets,
            projection_indices: projection_indices,
            cache: Mutex::new(HashMap::new()),
            max_cached: max_cached
        }
    }

    fn locate(&self, global_row: usize) -> (usize, usize) {
        let rg = match self.rg_offset.binary_search(&global_row) {
            Ok(i) => i,
            Err(i) => i - 1,
        };

        (rg, global_row - self.rg_offset[rg])
    }

    fn load_row_group(&self, rg_index: usize) -> Chunk {
        let file = File::open(&self.path).expect(&format!("Fail to open file {}", self.path));
        let metadata = parquet::file::metadata::ParquetMetaDataReader::new().parse_and_finish(&file).expect("Fail to read the metadata");
        let rg_rows = metadata.row_group(rg_index).num_rows() as usize;
        let schema_descr = metadata.file_metadata().schema_descr();

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("Fail to read")
            .with_projection(ProjectionMask::leaves(
                schema_descr, 
                self.projection_indices.clone())
            )
            .with_row_groups(vec![rg_index])
            .with_batch_size(rg_rows);

        let reader = builder.build().expect("Fail to build");

        let num_features = self.features.len();
        let mut features : Vec<Vec<f32>>= vec![Vec::with_capacity(rg_rows); num_features];
        let mut lables: Vec<f32> = Vec::with_capacity(rg_rows);

        for batch in reader {
            let batch = batch.expect("Fail to get batch");
            for (i, col_name) in self.features.iter().enumerate() {
                let array = batch.column_by_name(col_name).unwrap();
                let f32_arr = cast(array, &arrow::datatypes::DataType::Float32).unwrap();
                let vals: &PrimitiveArray<Float32Type> = f32_arr.as_primitive();
                features[i].extend_from_slice(vals.values());
            }

            let array = batch.column_by_name(&self.lables).unwrap();
            let f32_arr = cast(array, &arrow::datatypes::DataType::Float32).unwrap();
            let vals: &PrimitiveArray<Float32Type> = f32_arr.as_primitive();
            lables.extend_from_slice(vals.values());
        }

        Chunk {
            features: features,
            labels: lables,
        }
    }

    fn try_get(&self, rg_index: usize) -> MutexGuard<'_, HashMap<usize, Chunk>>{
        let mut cache = self.cache.lock().unwrap();

        if !cache.contains_key(&rg_index) {
            if cache.len() == self.max_cached {
                cache.clear();
            }

            let chunk = self.load_row_group(rg_index);
            cache.insert(rg_index, chunk);
        }

        cache
    }

    pub fn features_num(&self) -> usize {
        self.features.len()
    }

    pub fn split(
        path: &str,
        features: Vec<String>,
        labels: String,
        max_cached: usize,
        ratio: f32,
    ) -> (Self, Self) {
        let df = LazyFrame::scan_parquet(path.into(), ScanArgsParquet::default()).unwrap();

        let length = df.clone().select(&[len()]).collect().unwrap().column("len").unwrap().u32().unwrap().get(0).unwrap();
        let train_len= (length as f32 * ratio) as u32;

        let train_lf = df.clone().slice(0, train_len);
        let valid_lf = df.clone().slice(train_len as i64, length - train_len);

        let train_path = path.replace(".parquet", "_train.parquet");
        let valid_path = path.replace(".parquet", "_valid.parquet");

        if let Err(e) = sink_to_local(train_lf, &PathBuf::from(train_path.clone())) {
            eprintln!("Error saving the training data: {}", e);
        };

        if let Err(e) = sink_to_local(valid_lf, &PathBuf::from(valid_path.clone())) {
            eprintln!("Error saving the valid data: {}", e);
        };

        (
            ParquetDataset::new(&train_path, features.clone(), labels.clone(), max_cached),
            ParquetDataset::new(&valid_path, features, labels, max_cached)
        )
    }
}

impl Dataset<ParquetItem> for ParquetDataset {
    fn get(&self, index: usize) -> Option<ParquetItem> {
        if index > self.ttl_rows {
            return None;
        }

        let (rg_index, offset) = self.locate(index);

        let cache = self.try_get(rg_index);

        let chunk = cache.get(&rg_index).unwrap();
        let features: Vec<f32> = chunk.features.iter().map(|col| col[offset]).collect();

        Some(
            ParquetItem {
                feature: features,
                label: chunk.labels[offset],
            }
        )
    }

    fn len(&self) -> usize {
        self.ttl_rows
    }
}

