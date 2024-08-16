use burn::backend::autodiff::grads::Gradients;
use burn::{
    data::dataloader::{batcher::Batcher, DataLoaderBuilder},
    module::AutodiffModule,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{CpuTemperature, LossMetric},
        LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
};
use serde::{Deserialize, Serialize};

// Define the model configuration
#[derive(Config)]
pub struct PoincareTaxonomyEmbeddingModelConfig {
    pub taxonomy_size: usize,
    pub embedding_size: usize,
    // pub layer_norm_eps: f64,
}

// Define the model structure
#[derive(Module, Debug)]
pub struct PoincareTaxonomyEmbeddingModel<B: Backend> {
    embedding_token: Embedding<B>,
    // layer_norm: LayerNorm<B>,
}

// Define functions for model initialization
impl PoincareTaxonomyEmbeddingModelConfig {
    /// Initializes a model with default weights
    pub fn init<B: Backend>(&self, device: &B::Device) -> PoincareTaxonomyEmbeddingModel<B> {
        let initializer = burn::nn::Initializer::Uniform {
            min: 1e-8,
            max: 1e-3,
        };

        //let layer_norm = LayerNormConfig::new(self.embedding_size)
            //.with_epsilon(self.layer_norm_eps)
            //.init(device);

        let embedding_token = EmbeddingConfig::new(self.taxonomy_size, self.embedding_size)
            .with_initializer(initializer)
            .init(device);

        PoincareTaxonomyEmbeddingModel {
            embedding_token,
            // layer_norm,
        }
    }
}

impl<B: Backend> PoincareTaxonomyEmbeddingModel<B> {
    // Defines forward pass for training
    pub fn forward(&self, 
                origins: Tensor<B, 2, Int>, 
                branches: Tensor<B, 2, Int>) -> Tensor<B, 2, Float> {

        let dims = branches.dims(); // Should be 32, but let's make it dynamic

        // Calculate the Poincaré distance
        let origins = self
            .embedding_token
            .forward(origins);
        
        let destinations = self
            .embedding_token
            .forward(branches);

        // println!("{:#?}", x);
        // println!("{:#?}", y);

        // let y = self.layer_norm.forward(y);
        // let x = self.layer_norm.forward(x);

        poincare_distance(origins, destinations)
        /*

        // Simple euclidian
        let x = self
            .embedding_token
            .forward(pairs.clone().slice([0..pairs.dims()[0], 0..1]));

        let y = self
            .embedding_token
            .forward(pairs.clone().slice([0..pairs.dims()[0], 1..2]));

        // let x = self.layer_norm.forward(x);
        // let y = self.layer_norm.forward(y);
        // let x = x.add_scalar(1e-8);
        // let y = y.add_scalar(1e-8);

        let distance = (x - y).powf_scalar(2.0).sum_dim(2).sqrt();
        distance.squeeze(1)  */
    }

    pub fn forward_regression(
        &self,
        origins: Tensor<B, 2, Int>,
        pairs: Tensor<B, 2, Int>,
        distances: Tensor<B, 2, Float>,
    ) -> RegressionOutput<B> {
        let predicted_distances = self.forward(origins, pairs);

        let loss = (predicted_distances.clone() - distances.clone())
            .powf_scalar(2.0)
            .mean();

        RegressionOutput::new(loss, predicted_distances, distances)
    }
}

#[derive(Clone, Debug)]
pub struct TaxaDistance<const N: usize> {
    pub origin: u32,
    pub branches: [u32; N],
    pub distances: [u32; N],
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
    pub origins: Tensor<B, 2, Int>,
    pub branches: Tensor<B, 2, Int>,
    pub distances: Tensor<B, 2, Float>,
}

impl<B: Backend, const N: usize> Batcher<TaxaDistance<N>, TangoBatch<B>> for TangoBatcher<B> {
    fn batch(&self, items: Vec<TaxaDistance<N>>) -> TangoBatch<B> {

        let origins = items
            .iter()
            .map(|item| TensorData::from([item.origin]))
            .map(|data| Tensor::<B, 2, Int>::from_data(data.convert::<u32>(), &self.device))
            .collect();

        let branches = items
            .iter()
            .map(|item| TensorData::from(item.branches))
            .map(|data| Tensor::<B, 1, Int>::from_data(data.convert::<u32>(), &self.device))
            .map(|tensor| tensor.reshape([1, 2]))
            .collect();

        let distances = items
            .iter()
            .map(|item| TensorData::from([item.distances]))
            .map(|data| Tensor::<B, 1>::from_data(data.convert::<u32>(), &self.device))
            .map(|tensor| tensor.reshape([1, 1]))
            .collect();

        let branches = Tensor::cat(branches, 0).to_device(&self.device);
        let distances = Tensor::cat(distances, 0).to_device(&self.device);
        let origins = Tensor::cat(origins, 0).to_device(&self.device);

        TangoBatch {
            origins,
            branches,
            distances,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<TangoBatch<B>, RegressionOutput<B>>
    for PoincareTaxonomyEmbeddingModel<B>
{
    fn step(&self, batch: TangoBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch.origins, batch.branches, batch.distances);

        TrainOutput::new(self, item.loss.backward(), item)
    }
    
}

impl<B: Backend> ValidStep<TangoBatch<B>, RegressionOutput<B>>
    for PoincareTaxonomyEmbeddingModel<B>
{
    fn step(&self, batch: TangoBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(batch.origins, batch.branches, batch.distances)
    }
}

pub fn poincare_distance<B: Backend>(x: Tensor<B, 3>, y: Tensor<B, 3>) -> Tensor<B, 2> {
    // let device = x.device();

    // Broadcast x to the same shape as y
    let x = x.expand(y.dims());

    let x_norm = l2_norm(x.clone()); //.clamp(1e-5, 1.0 - 1e-5);
    let y_norm = l2_norm(y.clone()); // .clamp(1e-5, 1.0 - 1e-5);

    let diff = x - y;
    let diff_norm = l2_norm(diff).powf_scalar(2.0);

    let num = diff_norm * 2.0;
    let num = num.add_scalar(1e-8);
    let ones = Tensor::<B, 2>::ones_like(&x_norm);
    let denom =
        (ones.clone() - x_norm.clone().powf_scalar(2.0)) * (ones - y_norm.clone().powf_scalar(2.0));
    let denom = denom.add_scalar(1e-8);

    let distance = num / denom;

    let distance = distance.clamp(1.0, 1e8);

    acosh(distance)
}

pub fn l2_norm<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 2> {
    x.powf_scalar(2.0).sum_dim(2).sqrt().squeeze(2)
}

fn acosh<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    // let x = x.clamp_min(1.000005);
    let x_squared = x.clone().powf_scalar(2.0);
    let inside_sqrt = x_squared.sub_scalar(1.0);
    let sqrt_term = inside_sqrt.sqrt();
    (x + sqrt_term).log()
}

// Training stuff
#[derive(Config)]
pub struct TrainingConfig {
    pub model: PoincareTaxonomyEmbeddingModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 1024)]
    pub num_epochs: usize,
    #[config(default = 8)]
    pub batch_size: usize,
    #[config(default = 2)]
    pub num_workers: usize,
    #[config(default = 1337)]
    pub seed: u64,
    #[config(default = 1.0e-6)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Wgpu, autodiff::Autodiff};
    use burn::tensor::Tensor;

    #[test]
    fn test_poincare_distance() {
        type MyBackend = Wgpu<f32, i32>;
        let device = burn::backend::wgpu::WgpuDevice::default();

        let x = Tensor::<MyBackend, 3, Float>::from_data(
            [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]],
            &device,
        );
        let y = Tensor::<MyBackend, 3, Float>::from_data(
            [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]],
            &device,
        );

        let distance = poincare_distance(x, y);

        let expected =
            Tensor::<MyBackend, 2, Float>::from_data([[1.31696], [1.31696], [1.31696]], &device);

        // println!("{}", distance);
        // println!("{}", expected);

        let equals = distance.equal(expected).all();
        // let equals = equals.into_data().value[0];

        //assert!(equals);
    }
}
