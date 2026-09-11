use burn::{config::Config, data::dataloader::DataLoaderBuilder, module::{AutodiffModule, Module}, optim::{AdamConfig, GradientsParams, Optimizer}, record::CompactRecorder, tensor::{ElementConversion, backend::AutodiffBackend}};

use crate::{batcher::{ParquetBatch, ParquetBatcher}, model::{Mlp, MlpConfig}, parquet_dataset::ParquetDataset};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub feature_cols: Vec<String>,
    pub label_col: String,
    #[config(default = 128)]
    pub hidden_dim: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    #[config(default = 0.8)]
    pub train_ratio: f32,
    #[config(default = 50000)]
    pub chunk_size: usize,
    #[config(default = 4)]
    pub max_cached_chunks: usize,
}

pub fn train<B: AutodiffBackend>(
    path: &str,
    save_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) {
    B::seed(&device, config.seed);
    let num_features = config.feature_cols.len();

    let (train_dataset, valid_dataset) = ParquetDataset::split(path, config.feature_cols, config.label_col, config.max_cached_chunks, config.train_ratio);

    let train_batcher = ParquetBatcher::<B>::new(device.clone());
    let valid_batcher = ParquetBatcher::<B::InnerBackend>::new(device.clone());

    let train_dataloader = DataLoaderBuilder::new(train_batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let valid_dataloader = DataLoaderBuilder::new(valid_batcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    let model_config = MlpConfig::new(num_features, config.hidden_dim);
    let mut model: Mlp<B> = model_config.init(&device);
    let mut optim = AdamConfig::new().init::<B, Mlp<B>>();

    for epoch in 0..config.num_epochs {
        let mut train_loss = 0.0;
        let mut train_batches = 0;

        for batch in train_dataloader.iter() {
            let ParquetBatch {input, output} = batch;
            let loss = model.forward_loss(input, output);
            let loss_val: f64 = loss.clone().into_scalar().elem();

            let grad = loss.backward();
            let grad = GradientsParams::from_grads(grad, &model);
            model = optim.step(config.learning_rate, model, grad);

            train_loss += loss_val;
            train_batches += 1;
        }

        let train_avg_loss = train_loss / train_batches as f64;

        let mut valid_loss: f64 = 0.0;
        let mut valid_batches = 0;
        let valid_model = model.valid();

        for batch in valid_dataloader.iter() {
            let ParquetBatch {input, output} = batch;
            let loss = valid_model.forward_loss(input, output);

            valid_loss += loss.clone().into_scalar().elem::<f64>();
            valid_batches += 1;
        }

        let valid_avg_loss = valid_loss / valid_batches as f64;

        println!(
            "Epoch {}/{} — train_loss: {:.6}, valid_loss: {:.6}",
            epoch, config.num_epochs, train_avg_loss, valid_avg_loss
        );
    }

    let save_path = format!("{}/trained_model", save_dir);
    model.save_file(&save_path, &CompactRecorder::new()).expect("Fail to save the params");

}
