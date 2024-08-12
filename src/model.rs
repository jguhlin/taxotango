use burn::{
    backend::Wgpu,
    data::dataloader::{batcher::Batcher, DataLoaderBuilder},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::{activation::softmax, backend::AutodiffBackend},
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
};
use serde::{Deserialize, Serialize};

use crate::build_taxonomy_graph_gen;

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
        let x = self
            .embedding_token
            .forward(pairs.clone().slice([0..dims[0], 0..1]));
        let y = self
            .embedding_token
            .forward(pairs.slice([0..dims[0], 1..2]));

        // println!("X: {}", x);

        // let dims = x.dims();
        // println!("Embedding Dims: {:?}", dims);

        // let x = x.squeeze(0);
        // let y = y.squeeze(0);

        // println!("Getting distance");

        poincare_distance(x, y)
    }

    pub fn forward_regression(
        &self,
        pairs: Tensor<B, 2, Int>,
        distances: Tensor<B, 2, Float>,
    ) -> RegressionOutput<B> {
        let predicted_distances = self.forward(pairs);
        let loss = (predicted_distances.clone() - distances.clone())
            .powf_scalar(2.0)
            .mean();

        RegressionOutput::new(loss, predicted_distances, distances)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TaxaDistance {
    pub branches: [u32; 2],
    pub distance: f32,
}

#[derive(Clone)]
pub struct TangoBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> TangoBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct TangoBatch<B: Backend> {
    pub branches: Tensor<B, 2, Int>,
    pub distances: Tensor<B, 2, Float>,
}

impl<B: Backend> Batcher<TaxaDistance, TangoBatch<B>> for TangoBatcher<B> {
    fn batch(&self, items: Vec<TaxaDistance>) -> TangoBatch<B> {
        let branches = items
            .iter()
            .map(|item| Data::<u32, 1>::from(item.branches))
            .map(|data| Tensor::<B, 1, Int>::from_data(data.convert(), &self.device))
            .map(|tensor| tensor.reshape([1, 2]))
            .collect();

        let distances = items
            .iter()
            .map(|item| Data::<f32, 1>::from([item.distance]))
            .map(|data| Tensor::<B, 1>::from_data(data.convert(), &self.device))
            .map(|tensor| tensor.reshape([1, 1]))
            .collect();

        let branches = Tensor::cat(branches, 0).to_device(&self.device);
        let distances = Tensor::cat(distances, 0).to_device(&self.device);

        TangoBatch {
            branches,
            distances,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<TangoBatch<B>, RegressionOutput<B>>
    for PoincareTaxonomyEmbeddingModel<B>
{
    fn step(&self, batch: TangoBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch.branches, batch.distances);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<TangoBatch<B>, RegressionOutput<B>>
    for PoincareTaxonomyEmbeddingModel<B>
{
    fn step(&self, batch: TangoBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(batch.branches, batch.distances)
    }
}

pub fn poincare_distance<B: Backend>(x: Tensor<B, 3>, y: Tensor<B, 3>) -> Tensor<B, 2> {
    let device = x.device();

    let x_norm = l2_norm(x.clone()).clamp(0.0, 1.0 - 1e-5);
    let y_norm = l2_norm(y.clone()).clamp(0.0, 1.0 - 1e-5);

    let diff = x - y;
    // println!("Diff: {}", diff);
    // let diff = diff.powf(Tensor::<B, 3>::from_floats([[[2.0]]], &device));
    let diff = l2_norm(diff.clone());
    let diff = diff.powf_scalar(2.0);

    let num = diff * 2.0;
    let one = Tensor::<B, 2>::ones([1, 1], &device); // Create a tensor with value 1.0
    let denom = (one.clone() - x_norm.clone().powf_scalar(2.0))
        * (one.clone() - y_norm.clone().powf_scalar(2.0));
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
    // let dims = data.dims();
    // println!("L2 Norm Dims: {:?}", dims);
    // println!("L2 Norm Data: {}", data);
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

// Training stuff
#[derive(Config)]
pub struct TrainingConfig {
    pub model: PoincareTaxonomyEmbeddingModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 1337)]
    pub seed: u64,
    #[config(default = 1.0e-5)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    batch_gen: crate::BatchGenerator,
    device: B::Device,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train: TangoBatcher<B> = TangoBatcher::<B>::new(device.clone());
    let batcher_valid = TangoBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(batch_gen.train());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(batch_gen.test());

    log::info!("Creating learner");

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    log::trace!("Learner built");

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    log::trace!("Model trained");

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
