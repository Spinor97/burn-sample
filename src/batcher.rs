use burn::{Tensor, tensor::backend::Backend};

#[derive(Debug, Clone)]
pub struct ParquetBatch<B: Backend> {
    input: Tensor<B, 2>,
    output: Tensor<B, 1>,
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