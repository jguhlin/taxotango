use burn::{prelude::*, nn::{Embedding, EmbeddingConfig}};
use burn::train::metric::{Adaptor, LossInput};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn::nn::loss::*;

use super::poincare_distance;
use super::l2_norm;

const EPS: f32 = 1e-8;

// Define the model configuration
#[derive(Config)]
pub struct PoincareEmbeddingModelConfig {
    pub taxonomy_size: usize,
    pub embedding_size: usize,
}

// Define the model structure
#[derive(Module, Debug)]
pub struct PoincareEmbeddingModel<B: Backend> {
    pub embedding_token: Embedding<B>,
}

// Define functions for model initialization
impl PoincareEmbeddingModelConfig {
    /// Initializes a model with default weights
    pub fn init<B: Backend>(&self, device: &B::Device) -> PoincareEmbeddingModel<B> {
        let initializer = burn::nn::Initializer::Uniform {
            min: -0.015,
            max: 0.015,
        };

        let embedding_token = EmbeddingConfig::new(self.taxonomy_size, self.embedding_size)
            .with_initializer(initializer)
            .init(device);

            PoincareEmbeddingModel {
            embedding_token,
        }
    }
}

impl<B: Backend> PoincareEmbeddingModel<B> {
    // Defines forward pass for training
    pub fn forward(
        &self,
        embeddings: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3, Float> {
        let embeddings = self.embedding_token.forward(embeddings);

        embeddings
    }

    pub fn forward_distance(
        &self,
        origins: Tensor<B, 2, Int>,
        nearby: Tensor<B, 2, Int>,
        distant: Tensor<B, 2, Int>,
    ) -> EmbeddingOutput<B> {
        let origins_e = self.forward(origins);
        let nearby_e = self.forward(nearby);
        let distant_e = self.forward(distant);

        let positive_distances = poincare_distance(origins_e.clone(), nearby_e);
        let negative_distances = poincare_distance(origins_e.clone(), distant_e);

        let positive_loss = positive_distances.clone().sum_dim(1);
        let negative_loss = negative_distances.clone().sum_dim(1).clamp_max(1e-6);

        let loss = positive_loss + negative_loss;
        let loss = loss.mean();

        EmbeddingOutput::new(loss)
    }
}

#[derive(Module, Clone, Debug)]
pub struct PoincareLoss;

impl Default for PoincareLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl PoincareLoss {
    /// Create the criterion.
    pub fn new() -> Self {
        Self
    }

    /// Compute the criterion on the input tensor.
    ///
    /// # Shapes
    ///
    /// - logits: [batch_size, num_targets]
    /// - targets: [batch_size, num_targets]
    pub fn forward<B: Backend>(
        &self,
        origins: Tensor<B, 3>,
        nearby: Tensor<B, 3>,
        distant: Tensor<B, 3>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let tensor = self.forward_no_reduction(origins, nearby, distant);
        match reduction {
            Reduction::Mean | Reduction::Auto => tensor.mean(),
            Reduction::Sum => tensor.sum(),
        }
    }

    pub fn forward_no_reduction<B: Backend>(
        &self,
        origins: Tensor<B, 3>,
        nearby: Tensor<B, 3>,
        distant: Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        let positive_distances = poincare_distance(origins.clone(), nearby); // Shape: [batch_size, n]
        let negative_distances = poincare_distance(origins.clone(), distant); // Shape: [batch_size, m]

        let numerator = positive_distances.clone().neg().exp(); // e^-d(origins, nearby)
        let denominator = negative_distances.clone().neg().exp().sum_dim(1).clamp_min(EPS); // Σ e^-d(origins, distant)

        let loss = numerator / denominator.add_scalar(EPS);

        loss.log()
    }

    /*
    pub fn forward_no_reduction<B: Backend>(
        &self,
        origins: Tensor<B, 3>,
        nearby: Tensor<B, 3>,
        distant: Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        let positive_distances = poincare_distance(origins.clone(), nearby); // Shape: [batch_size, n]
        let negative_distances = poincare_distance(origins.clone(), distant); // Shape: [batch_size, m]
    
        // Step 1: Compute negative of distances
        let z_p = positive_distances.clone().neg(); // Shape: [batch_size, n]
        let z_n = negative_distances.clone().neg(); // Shape: [batch_size, m]
    
        // Step 2: Concatenate z_p and z_n
        // let z_all = z_p.clone().cat(z_n.clone(), 1); // Shape: [batch_size, n + m]
        let z_all = burn::tensor::Tensor::cat(vec![z_p.clone(), z_n.clone()], 1); // Shape: [batch_size, n + m]
    
        // Step 3: Compute max_z
        let max_z = z_all.clone().max_dim(1); // Shape: [batch_size, 1]
    
        // Step 4: Adjust z_p and z_n
        let adjusted_z_p = z_p.clone().sub(max_z.clone());
        let adjusted_z_n = z_n.clone().sub(max_z.clone());
    
        // Step 5: Compute exponentials
        let exp_adjusted_z_p = adjusted_z_p.exp(); // Shape: [batch_size, n]
        let exp_adjusted_z_n = adjusted_z_n.exp(); // Shape: [batch_size, m]
    
        // Step 6: Sum exponentials
        let sum_exp = exp_adjusted_z_p.sum_dim(1).add(exp_adjusted_z_n.sum_dim(1)); // Shape: [batch_size, 1]
    
        // Step 7: Compute log-sum-exp
        let log_sum_exp = max_z.clone().add(sum_exp.log()); // Shape: [batch_size, 1]
    
        // Step 8: Compute final loss
        let loss = positive_distances.clone().sum_dim(1).add(log_sum_exp); // Shape: [batch_size, 1]
    
        loss 
    } */
    

    /*
    /// Compute the criterion on the input tensor without reducing.
    pub fn forward_no_reduction<B: Backend>(
        &self,
        origins: Tensor<B, 3>,
        nearby: Tensor<B, 3>,
        distant: Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        let positive_distances = poincare_distance(origins.clone(), nearby);
        let negative_distances = poincare_distance(origins.clone(), distant);

        // println!("Positive Distances: {}", positive_distances);
        // println!("Negative Distances: {}", negative_distances);

        // let positive_loss = positive_distances.clone().sum_dim(1);
        // let negative_loss = negative_distances.clone().sum_dim(1).clamp_max(1e-6);

        // println!("Positive Loss: {}", positive_loss);
        // println!("Negative Loss: {}", negative_loss);

        // Log-sum-exp trick for numerical stability
        let max_neg_distance = negative_distances.clone().max_dim(1);
        // println!("Max Negative Distance: {}", max_neg_distance);
        let sum_neg_distances = (negative_distances.clone().neg().add(max_neg_distance)).exp().sum_dim(1);
        // println!("Sum Negative Distances: {}", sum_neg_distances);

        // Calculate log-loss for this pair
        let log_loss = (positive_distances.clone().neg().exp()) / (positive_distances.clone().neg().exp() + sum_neg_distances);
        // println!("Log Loss: {}", log_loss);
        let loss = log_loss.log().neg();
        // println!("Loss: {}", loss);

        // let loss = positive_loss + negative_loss.neg().add_scalar(1e-4);
        // println!("Loss: {}", loss);
        loss
    } */
}


#[derive(Module, Debug)]
pub struct EmbeddingOutput<B: Backend> {
    /// The loss.
    pub loss: Tensor<B, 1>,
}

impl<B: Backend> EmbeddingOutput<B> {
    /// Creates a new [EmbeddingOutput](EmbeddingOutput).
    pub fn new(loss: Tensor<B, 1>) -> Self {
        Self { loss }
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for EmbeddingOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

pub fn retraction<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    // If any have norms > 1, retract them back
    let mut x = x.clone();

    let mut norms = l2_norm(x.clone());
    let mut gte = norms.clone().greater_equal_elem(1.0);

    while gte.clone().any().into_scalar() {
        println!("Retracting...");
        println!("x: {}", x);
        println!("Norms: {}", norms);
        let replaced = x.clone() / norms;
        println!("Replaced: {}", replaced);
        let replaced = replaced.sub_scalar(1e-8);
        println!("Replaced: {}", replaced);
        let gte_mask = gte.clone().expand(x.shape());
        println!("GTE Mask: {}", gte_mask);
        x = x.mask_where(gte_mask, replaced);
        println!("New x: {}", x);
        norms = l2_norm(x.clone());
        gte = norms.clone().greater_equal_elem(1.0);
    }

    x

}