use burn::{
    backend::Wgpu,
    nn::{
        loss::CrossEntropyLossConfig,
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    prelude::*,
    tensor::{activation::softmax, backend::AutodiffBackend},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

// Define the model configuration
#[derive(Config)]
pub struct PoincareTaxonomyEmbeddingModelConfig {
    taxonomy_size: usize,
    embedding_size: usize,
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

        PoincareTaxonomyEmbeddingModel {
            embedding_token,
        }
    }
}

impl<B: Backend> PoincareTaxonomyEmbeddingModel<B> {
    // Defines forward pass for training
    pub fn forward(&self, pairs: Tensor<B, 2, Int>) -> f16 {
        // Get the embeddings for the tokens
        let token_embeddings = self.embedding_token.forward(pairs);

        // Calculate the Poincaré distance
        let u = token_embeddings.get(0);
        let v = token_embeddings.get(1);
        let distance = poincare_distance(u, v);

        // Return the distance
        distance.get(0)
        
    }
}

fn poincare_distance<B: Backend>(u: &Tensor<B, 1>, v: &Tensor<B, 1>) -> Tensor<B, 1> {
    let epsilon = 1e-5;
    
    // Calculate norms
    let norm_u = u.clone().norm(1, None, None).clamp(0.0, 1.0 - epsilon);
    let norm_v = v.clone().norm(1, None, None).clamp(0.0, 1.0 - epsilon);
    
    // Calculate the squared difference
    let diff = u.clone() - v.clone();
    let squared_diff_norm = diff.clone().norm(2, None, None).powf(2.0);
    
    // Calculate the hyperbolic distance
    let num = squared_diff_norm * 2.0;
    let denom = (1.0 - norm_u.clone().powf(2.0)) * (1.0 - norm_v.clone().powf(2.0));
    let distance = num / denom;
    
    // Apply acosh to get the Poincaré distance
    let result = acosh(distance);
    
    result
}