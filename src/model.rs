use burn::{Tensor, config::Config, module::Module, nn::{Linear, LinearConfig, Relu}, tensor::backend::Backend};

#[derive(Config, Debug)]
pub struct MlpConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    #[config(default = 1)]
    pub output_dim: usize,
    #[config(default = 0.5)]
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    relu: Relu
}

impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        Mlp {
            linear1: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            linear2: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            linear3: LinearConfig::new(self.hidden_dim, self.output_dim).init(device),
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> Mlp<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.relu.forward(self.linear1.forward(x));
        let x = self.relu.forward(self.linear2.forward(x));

        self.linear3.forward(x)
    }

    pub fn forward_loss(&self, x: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
        let output = self.forward(x);
        let diff = output - targets;
        (diff.clone() * diff).mean()
    }
}

