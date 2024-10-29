use burn::nn::loss::*;
use burn::backend::{autodiff::Autodiff, Wgpu};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn::train::metric::{Adaptor, LossInput};
use burn::{
    nn::{Embedding, EmbeddingConfig},
    prelude::*,
};

use std::sync::Arc;
use std::collections::HashMap;

use super::l2_norm;
use super::poincare_distance;

const EPS: f32 = 1e-18;
const EPS_LARGE: f32 = 1e-17;

// Define the model configuration
#[derive(Config)]
pub struct PoincareEmbeddingModelConfig {
    pub taxonomy_size: usize,
    pub embedding_size: usize,
    pub root_idx: usize,
}

// Define the model structure
#[derive(Module, Debug)]
pub struct PoincareEmbeddingModel<B: Backend> {
    pub embedding_token: Embedding<B>,
    pub root_idx: usize,
    pub epoch: u64,
    pub warmup: bool,
    pub sample_weights: Tensor<B, 2, Float>,
}

// Define functions for model initialization
impl PoincareEmbeddingModelConfig {
    /// Initializes a model with default weights
    pub fn init<B: AutodiffBackend>(&self, devices: Vec<B::Device>) -> PoincareEmbeddingModel<B>  
     {

        let device = &devices[0];

        let initializer = burn::nn::Initializer::Uniform {
            min: -0.5,
            max: 0.5,
        };

        let embedding_token = EmbeddingConfig::new(self.taxonomy_size, self.embedding_size)
            .with_initializer(initializer)
            .init(device);

        PoincareEmbeddingModel {
            embedding_token,
            root_idx: self.root_idx,
            epoch: 0,
            warmup: true,
            sample_weights: Tensor::ones([self.taxonomy_size, 1], device),
        }
    }
}

impl<B: Backend> PoincareEmbeddingModel<B> {
    // Defines forward pass for training
    pub fn forward(&self, embeddings: Tensor<B, 2, Int>) -> Tensor<B, 3, Float> {
        let embeddings = self.embedding_token.forward(embeddings);

        embeddings
    }

    pub fn forward_simple_distance(
        &self,
        parent: Tensor<B, 2, Int>,
        child: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        let parent_e = self.forward(parent);
        let child_e = self.forward(child);

        let distance = poincare_distance(parent_e, child_e);
        // println!("Distance: {}", distance);
        distance
    }

    pub fn gather_sample_weights(&self, indices: Tensor<B, 2, Int>) -> Tensor<B, 2, Float> {
        // Gather from self.sample_weights
        let sample_weights = self.sample_weights.clone().gather(2, indices);

        sample_weights
    }

    pub fn forward_distance(
        &self,
        origins: Tensor<B, 2, Int>,
        nearby: Tensor<B, 2, Int>,
        distant: Tensor<B, 2, Int>,
        origin_weight_factor: Tensor<B, 2, Float>,
        nearby_weight_factor: Tensor<B, 2, Float>,
        distant_weight_factor: Tensor<B, 2, Float>,
    ) -> EmbeddingOutput<B> {
        let huber_delta = 0.02;

        let device = origins.device();

        let origins_e = self.forward(origins.clone());
        // let origins_sample_weights = self.gather_sample_weights(origins);
        let nearby_e = self.forward(nearby.clone());
        // let nearby_sample_weights = self.gather_sample_weights(nearby);
        let distant_e = self.forward(distant.clone());
        // let distant_sample_weights = self.gather_sample_weights(distant);

        let num_samples = distant_e.shape().dims[1] as i64;
        let num_samples = (num_samples as f32 * 0.1).round() as i64;
        // Minimum 1
        let num_samples = if num_samples == 0 { 1 } else { num_samples };

        // If we are in the warmup phase, only use 10% of the negative samples
        let distant_e = if self.warmup {
            distant_e.slice([None, Some((0, num_samples))])
        } else {
            distant_e
        };

        // Same for the weights
        let distant_weight_factor = if self.warmup {
            distant_weight_factor.slice([None, Some((0, num_samples))])
        } else {
            distant_weight_factor
        };

        let positive_distances = poincare_distance(origins_e.clone(), nearby_e.clone());
        let negative_distances = poincare_distance(origins_e.clone(), distant_e.clone());

        // let positive_distances = positive_distances * nearby_weight_factor;
        // let negative_distances = negative_distances * distant_weight_factor;

        let numerator = positive_distances.clone().neg().exp(); // e^-d(origins, nearby)

        // Sum numerator
        let numerator_sum: Tensor<B, 1> = numerator.clone().sum_dim(1).squeeze(1); // Σ e^-d(origins, nearby)

        let denominator: Tensor<B, 1> = negative_distances
            .clone()
            .neg()
            .exp()
            .sum_dim(1)
            .squeeze(1); // Σ e^-d(origins, distant)
       
        let denominator = denominator
            .add(numerator_sum); // Σ e^-d(origins, distant)

        let numerator = if self.warmup {
            numerator.mul_scalar(1000.0)
        } else {
            numerator
        };

        let loss = numerator.clone().clamp_min(EPS).squeeze(1) / denominator.clamp_min(EPS_LARGE);
        // let loss = loss.clamp_min(EPS);
        let loss = loss.log().neg();

        // Root regularization
        let root_regularization = 0.001;
        let root_query = Tensor::<B, 2, Int>::from_ints([[self.root_idx as u64]], &device);
        let root: Tensor<B, 3> = self.forward(root_query);
        let root_norm_squared = root.powf_scalar(2.0).sum();
        let mut root_loss = root_norm_squared * root_regularization;
        root_loss = root_loss.clamp_min(EPS);
       
        // So norm of the origin should be the origin_weight_factor

        let loss = loss.add(root_loss);

        return EmbeddingOutput::new(loss.sum());
    }

    
}

#[derive(Module, Debug)]
pub struct EmbeddingOutput<B: Backend> {
    /// The loss.
    pub loss: Tensor<B, 1>,
}

impl<B: Backend> EmbeddingOutput<B> {
    /// Creates a new [EmbeddingOutput](EmbeddingOutput).
    pub fn new(loss: Tensor<B, 1>) -> Self {
        Self { loss: loss }
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for EmbeddingOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}
