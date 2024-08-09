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
    pub fn forward(&self, pairs: Tensor<B, 2, Int>) -> Tensor<B, 2, Float> {
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

pub fn poincare_distance<B: Backend>(x: Tensor<B, 3>, y: Tensor<B, 3>) -> Tensor<B, 2> {
    let device = x.device();

    let x_norm = l2_norm(x.clone()).clamp(0.0, 1.0 - 1e-5);
    let y_norm = l2_norm(y.clone()).clamp(0.0, 1.0 - 1e-5);

    let diff = x - y;
    println!("Diff: {}", diff);
    // let diff = diff.powf(Tensor::<B, 3>::from_floats([[[2.0]]], &device));
    let diff = l2_norm(diff.clone());
    let diff = diff.powf_scalar(2.0);

    let num = diff * 2.0;
    let one = Tensor::<B, 2>::ones([1, 1], &device); // Create a tensor with value 1.0
    let denom = (one.clone() - x_norm.clone().powf_scalar(2.0)) * (one.clone() - y_norm.clone().powf_scalar(2.0));
    let distance = num / denom;

    let result = acosh(distance);

    result
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
    // let device = x.device();
    let data = x.powf_scalar(2.0).sum_dim(2).sqrt();
    let dims = data.dims();
    println!("L2 Norm Dims: {:?}", dims);
    println!("L2 Norm Data: {}", data);
    data.squeeze(1)
}

fn acosh<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    // let device = x.device();

    // Compute x^2
    let x_squared = x.clone().powf_scalar(2.0);
    
    // Compute the inside of the square root: x^2 - 1
    let inside_sqrt = x_squared.clone() - Tensor::<B, 2>::ones_like(&x_squared);
    
    // Compute the square root
    let sqrt_term = inside_sqrt.sqrt();
    
    // Compute x + sqrt(x^2 - 1)
    let add_term = x + sqrt_term;
    
    // Compute ln(x + sqrt(x^2 - 1))
    let acosh_result = add_term.log();
    
    acosh_result
}
