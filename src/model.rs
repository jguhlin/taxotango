use burn::{
    backend::Wgpu,
    nn::{loss::CrossEntropyLossConfig, Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
    tensor::{activation::softmax, backend::AutodiffBackend},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

// Define the model configuration
#[derive(Config)]
pub struct PoincareTaxonomyEmbeddingModelConfig {
    pub taxonomy_size: usize,
    pub embedding_size: usize,
}

// Define the model structure
#[derive(Module, Debug)]
pub struct PoincareTaxonomyEmbeddingModel<B: Backend> {
    embedding_token: Embedding<B>,
}

// Define functions for model initialization
impl PoincareTaxonomyEmbeddingModelConfig {
    /// Initializes a model with default weights
    pub fn init<B: Backend>(&self, device: &B::Device) -> PoincareTaxonomyEmbeddingModel<B> {
        let embedding_token =
            EmbeddingConfig::new(self.taxonomy_size, self.embedding_size).init(device);

        PoincareTaxonomyEmbeddingModel { embedding_token }
    }
}

impl<B: Backend> PoincareTaxonomyEmbeddingModel<B> {
    // Defines forward pass for training
    pub fn forward(&self, pairs: Tensor<B, 2, Int>) -> Tensor<B, 3, Float> {
        let dims = pairs.dims();
        // println!("{:?}", dims);
        // println!("{}", pairs);
        // println!("Pairs sliced: {}", pairs.clone().slice([0..dims[0], 0..1]));

        // Calculate the Poincaré distance
        let x = self.embedding_token.forward(pairs.clone().slice([0..dims[0], 0..1]));
        let y = self.embedding_token.forward(pairs.slice([0..dims[0], 1..2]));

        // println!("X: {}", x);

        let dims = x.dims();
        println!("Embedding Dims: {:?}", dims);

        // let x = x.squeeze(0);
        // let y = y.squeeze(0);

        println!("Getting distance");

        poincare_distance(x, y)
    }
}

pub fn poincare_distance<B: Backend>(x: Tensor<B, 3>, y: Tensor<B, 3>) -> Tensor<B, 3> {
    let device = x.device();

    // let x_norm = l2_norm(x.clone()).clamp(0.0, 1.0 - 1e-5);
    // let y_norm = l2_norm(y.clone()).clamp(0.0, 1.0 - 1e-5);

    let diff = x - y;
    println!("Diff: {}", diff);
    // let diff = diff.powf(Tensor::<B, 3>::from_floats([[[2.0]]], &device));
    let diff = diff.powf_scalar(2.0);
    println!("Diff powf: {}", diff);
    // let diff = diff.sum_dim(3);

    // todo the rest, just seend what works right now though
    diff
}

/// Calculate the L2 norm of a tensor
///
/// # Arguments
/// * `x` - A tensor of shape [batch_size, embedding_size]
///
/// # Returns
/// A tensor of shape [batch_size] containing the L2 norm of each row in `x`
///
pub fn l2_norm<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 2> {
    let device = x.device();
    let data = x.powf_scalar(2.0).sum_dim(2).sqrt();
    let dims = data.dims();
    println!("L2 Norm Dims: {:?}", dims);
    println!("L2 Norm Data: {}", data);
    data.squeeze(1)
}
