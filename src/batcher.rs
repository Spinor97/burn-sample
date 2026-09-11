use burn::{Tensor, data::dataloader::batcher::Batcher, tensor::backend::Backend};

use crate::parquet_dataset::ParquetItem;

#[derive(Debug, Clone)]
pub struct ParquetBatch<B: Backend> {
    pub input: Tensor<B, 2>,
    pub output: Tensor<B, 2>,
}

#[derive(Clone)]
pub struct ParquetBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> ParquetBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            device: device,
        }
    }
}

impl<B: Backend> Batcher<B, ParquetItem, ParquetBatch<B>> for ParquetBatcher<B> {
    fn batch(&self, items: Vec<ParquetItem>, _device: &burn::prelude::Device<B>) -> ParquetBatch<B> {
        let batch_size = items.len();
        let num_features = items[0].feature.len();

        let flattened: Vec<f32> = items
            .iter()
            .flat_map(|item| item.feature.iter().copied())
            .collect();

        let input = Tensor::<B, 1>::from_floats(flattened.as_slice(), &self.device).reshape([batch_size, num_features]);
        let all_lables: Vec<f32> = items.iter().map(|item| item.label).collect();

        let output = Tensor::<B, 1>::from_floats(all_lables.as_slice(), &self.device).reshape([batch_size, 1]);

        ParquetBatch {
            input: input,
            output: output,
        }
    }
}