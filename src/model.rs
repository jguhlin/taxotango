use burn::backend::autodiff::grads::Gradients;
use burn::lr_scheduler;
use burn::record::Recorder;
use burn::train::metric::LearningRateMetric;
use burn::{
    data::dataloader::{batcher::Batcher, DataLoaderBuilder},
    module::AutodiffModule,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig},
    optim::{AdamConfig, AdamWConfig, GradientsParams, Optimizer, SgdConfig},
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{CpuTemperature, LossMetric},
        LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
};
use nn::loss::MseLoss;
use nn::{Linear, LinearConfig};
use serde::{Deserialize, Serialize};
use rerun::{demo_util::grid, external::glam};

// Define the model configuration
#[derive(Config)]
pub struct PoincareTaxonomyEmbeddingModelConfig {
    pub taxonomy_size: usize,
    pub embedding_size: usize,
    // pub layer_norm_eps: f64,
}

#[derive(Module, Debug, Clone)]
pub struct PoincareDistance {
    pub l2_norm: L2Norm,
}

impl PoincareDistance {
    pub fn new() -> Self {
        Self {
            l2_norm: L2Norm::new(),
        }
    }

    pub fn forward<B: Backend>(&self, x: Tensor<B, 3>, y: Tensor<B, 3>) -> Tensor<B, 2> {
        let x_norm = self.l2_norm.forward(x.clone()).clamp(1e-5, 1.0 - 1e-5);
        let y_norm = self.l2_norm.forward(y.clone()).clamp(1e-5, 1.0 - 1e-5);

        let diff_norm = self.l2_norm.forward(x - y).powf_scalar(2.0);

        let num = diff_norm * 2.0;
        let num = num.add_scalar(1e-7);
        let ones = Tensor::<B, 3>::ones_like(&x_norm);
        let denom = (ones.clone() - x_norm.clone().powf_scalar(2.0))
            * (ones - y_norm.clone().powf_scalar(2.0));
        let denom = denom.add_scalar(1e-7);

        let distance = num / denom;

        let distance = distance.squeeze(2);

        acosh(distance.mul_scalar(2).add_scalar(1.0))
    }
}

#[derive(Module, Debug, Clone)]
pub struct L2Norm {}

impl L2Norm {
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward<B: Backend, const N: usize>(&self, x: Tensor<B, N>) -> Tensor<B, N> {
        // Because you can't take the sqrt of
        x.powf_scalar(2.0).sum_dim(N - 1).add_scalar(1e-7).sqrt()
    }
}

// Define the model structure
#[derive(Module, Debug)]
pub struct PoincareTaxonomyEmbeddingModel<B: Backend> {
    embedding_token: Embedding<B>,
    l2_norm: L2Norm,
    poincare_distance: PoincareDistance,
    // scaling_inner: Linear<B>,
    scaling_layer: Linear<B>,
    // layer_norm: LayerNorm<B>,
}

// Define functions for model initialization
impl PoincareTaxonomyEmbeddingModelConfig {
    /// Initializes a model with default weights
    pub fn init<B: Backend>(&self, device: &B::Device) -> PoincareTaxonomyEmbeddingModel<B> {
        let initializer = burn::nn::Initializer::Uniform {
            min: -0.01,
            max: 0.01,
        };

        //let layer_norm = LayerNormConfig::new(self.embedding_size)
        //.with_epsilon(self.layer_norm_eps)
        //.init(device);

        let embedding_token = EmbeddingConfig::new(self.taxonomy_size, self.embedding_size)
            .with_initializer(initializer)
            .init(device);

        let scaling_layer = LinearConfig::new(1, 1).with_bias(false);

        PoincareTaxonomyEmbeddingModel {
            embedding_token,
            l2_norm: L2Norm::new(),
            poincare_distance: PoincareDistance::new(),
            scaling_layer: scaling_layer.init(device),
            // layer_norm,
        }
    }
}

impl<B: Backend> PoincareTaxonomyEmbeddingModel<B> {
    // Defines forward pass for training
    pub fn forward(
        &self,
        origins: Tensor<B, 2, Int>,
        branches: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2, Float> {
        // let dims = branches.dims(); // Should be 32, but let's make it dynamic

        // println!("{}", origins);
        // println!("{}", branches);

        let origins = self.embedding_token.forward(origins);
        let destinations = self.embedding_token.forward(branches);
        let origins = origins.expand(destinations.dims());

        // Calculate the Poincaré distance
        let distances = self.poincare_distance.forward(origins, destinations);
        // distances.mul_scalar(100.0)
        let distances: Tensor<B, 3> = distances.unsqueeze_dims(&[-1]);
        self.scaling_layer.forward(distances).squeeze(2)

        /*

        // Simple euclidian

        // let distance = (origins - destinations).sum_dim(2); // .powf_scalar(2.0).sum_dim(2).sqrt();
        // println!("{}", distance);
        let distance = self.l2_norm.forward(origins - destinations);
        let dims = distance.dims();
        distance.squeeze(2)  */
    }

    pub fn forward_regression(
        &self,
        origins: Tensor<B, 2, Int>,
        pairs: Tensor<B, 2, Int>,
        distances: Tensor<B, 2, Float>,
    ) -> RegressionOutput<B> {
        let predicted_distances = self.forward(origins, pairs);
        // log::debug!("Predicted distances: {}", predicted_distances);
        // log::debug!("Expected distances: {}", distances);

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
            .map(|data| Tensor::<B, 1, Int>::from_data(data.convert::<u32>(), &self.device))
            .map(|tensor| tensor.reshape([1, 1]))
            .collect();

        let branches = items
            .iter()
            .map(|item| TensorData::from(item.branches))
            .map(|data| Tensor::<B, 1, Int>::from_data(data.convert::<u32>(), &self.device))
            .map(|tensor| tensor.reshape([1, 8]))
            .collect();

        let distances = items
            .iter()
            .map(|item| TensorData::from([item.distances]))
            .map(|data| Tensor::<B, 2>::from_data(data.convert::<u32>(), &self.device))
            .map(|tensor| tensor.reshape([1, 8]))
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

fn acosh<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let x = x.clamp_min(1.0);
    let x_squared = x.clone().powf_scalar(2.0);
    let inside_sqrt = x_squared.sub_scalar(1.0);
    let sqrt_term = inside_sqrt.sqrt();
    (x + sqrt_term).log()
}

// Training stuff
#[derive(Config)]
pub struct TrainingConfig {
    pub model: PoincareTaxonomyEmbeddingModelConfig,
    // pub optimizer: AdamConfig,
    // pub optimizer: SgdConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 1024)]
    pub num_epochs: usize,
    #[config(default = 256)]
    pub batch_size: usize,
    #[config(default = 6)]
    pub num_workers: usize,
    #[config(default = 1337002)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<const D: usize, B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    mut batch_gen: crate::BatchGenerator<D>,
    device: B::Device,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train: TangoBatcher<B> = TangoBatcher::<B>::new(device.clone());
    let batcher_valid = TangoBatcher::<B::InnerBackend>::new(device.clone());

    let mut valid_ds = batch_gen.valid();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(batch_gen);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_ds);

    log::info!("Creating learner");

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            burn::lr_scheduler::linear::LinearLrSchedulerConfig::new(8e-3, 1e-6, 1_000_000).init(),
        );

    log::trace!("Learner built");

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    log::trace!("Model trained");

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}

pub fn custom_training_loop<const D: usize, B: AutodiffBackend>(
    batch_gen: crate::BatchGenerator<D>,
    device: &B::Device,
) {
    println!("Starting training loop");

    // let adamconfig = AdamConfig::new()
    // .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(1.0)));

    //let sgdconfig = SgdConfig::new()
    //.with_gradient_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(0.1)));

    let adamwconfig = AdamWConfig::new();
    // .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(1.0)));

    let config = PoincareTaxonomyEmbeddingModelConfig {
        taxonomy_size: batch_gen.taxonomy_size(),
        embedding_size: 16,
    };

    B::seed(1337);

    let config = TrainingConfig::new(config, adamwconfig);

    // Create the model and optimizer.
    let mut model: PoincareTaxonomyEmbeddingModel<B> = config.model.init(device);
    let mut optim = config.optimizer.init();

    if model.embedding_token.weight.contains_nan().into_scalar() {
        panic!("NaN detected in model, aborting - Did not even start!");
    }

    let batcher_train: TangoBatcher<B> = TangoBatcher::<B>::new(device.clone());
    let batcher_valid = TangoBatcher::<B::InnerBackend>::new(device.clone());

    let ds_valid = batch_gen.valid();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(batch_gen);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ds_valid);

    // Iterate over our training and validation loop for X epochs.
    for epoch in 1..config.num_epochs + 1 {
        // Implement our training loop.
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward(batch.origins, batch.branches);
            let loss = MseLoss::new().forward(
                output.clone(),
                batch.distances.clone(),
                burn::nn::loss::Reduction::Auto,
            );

            println!(
                "[Train - Epoch {} - Iteration {}] Loss {:.3}",
                epoch,
                iteration,
                loss.clone().into_scalar(),
            );

            // Gradients for the current backward pass
            let grads = loss.backward();

            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);

            // Get param ids
            let param_ids = burn::module::list_param_ids(&model);
            println!("Param ids: {:?}", param_ids);
            println!("Len: {} - Is Empty? {} ", grads.len(), grads.is_empty());
            //for id in param_ids {
            //let grad: Tensor<B, 1> = match grads.get(&id) {
            //Some(grad) => grad.clone(),
            //None => continue,
            //};

            //println!("{}", grad);
            //}

            // Update the model using the optimizer.
            model = optim.step(config.learning_rate, model, grads);

            // If any nan's detected, abort and print out most recent data
            if model.embedding_token.weight.contains_nan().into_scalar() {
                println!("Output: {}", output);
                println!("Distances: {}", batch.distances);

                panic!("NaN detected in model, aborting training - Pre normalization");
            }

            // let (id, weights) = model.embedding_token.weight.consume();

            // model.embedding_token.weight = burn::module::Param::initialized(id, normalize_to_poincare_ball(weights));
            // model.embedding_token.weight = model.embedding_token.weight.map(|x| normalize_to_poincare_ball(x));

            // If any nan's detected, abort and print out most recent data
            if model.embedding_token.weight.contains_nan().into_scalar() {
                println!("Output: {}", output);
                println!("Distances: {}", batch.distances);

                panic!("NaN detected in model, aborting training");
            }
        }

        // Get the model without autodiff.
        let model_valid = model.valid();

        // Implement our validation loop.
        for (iteration, batch) in dataloader_test.iter().enumerate() {
            let output = model_valid.forward(batch.origins, batch.branches);
            let loss = MseLoss::new().forward(
                output.clone(),
                batch.distances.clone(),
                burn::nn::loss::Reduction::Auto,
            );

            println!(
                "[Valid - Epoch {} - Iteration {}] Loss {}",
                epoch,
                iteration,
                loss.clone().into_scalar(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{autodiff::Autodiff, Wgpu};
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

    #[test]
    fn test_acosh() {
        type MyBackend = Wgpu<f32, i32>;
        let device = burn::backend::wgpu::WgpuDevice::default();

        let x = Tensor::<MyBackend, 3, Float>::from_data(
            [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]],
            &device,
        );
        let y = Tensor::<MyBackend, 3, Float>::from_data(
            [[[1.0, 1.0], [1.0, 1.0], [5.0, 5.0]]],
            &device,
        );

        let distance = poincare_distance(x, y);

        let expected =
            Tensor::<MyBackend, 2, Float>::from_data([[13.588997, 13.588997, 16.80787]], &device);

        let equals = distance.clone().equal(expected).all();
        let equals = equals.into_scalar();

        println!("{}", distance);

        assert!(equals);

        let x = Tensor::<MyBackend, 3, Float>::from_data(
            [[[0.005, 0.002], [0.01, 0.26], [0.51, 0.0872]]],
            &device,
        );
        let y = Tensor::<MyBackend, 3, Float>::from_data(
            [[[0.008, 0.004], [0.015, 0.268], [0.19, 0.00007]]],
            &device,
        );

        let distance = poincare_distance(x, y);

        let expected = Tensor::<MyBackend, 2, Float>::from_data(
            [[0.01022465, 0.028696949, 1.0654087]],
            &device,
        );

        let equals = distance.clone().equal(expected).all();
        let equals = equals.into_scalar();

        println!("{}", distance);

        assert!(equals);

        let x = Tensor::<MyBackend, 3, Float>::from_data(
            [[[-0.005, -0.002], [-0.01, -0.26], [-0.51, -0.0872]]],
            &device,
        );
        let y = Tensor::<MyBackend, 3, Float>::from_data(
            [[[0.008, 0.004], [0.015, 0.268], [0.19, 0.00007]]],
            &device,
        );

        let distance = poincare_distance(x, y);

        let expected =
            Tensor::<MyBackend, 2, Float>::from_data([[0.04050305, 1.4711125, 2.0157838]], &device);

        let equals = distance.clone().equal(expected).all();
        let equals = equals.into_scalar();

        println!("{}", distance);

        assert!(equals);
    }
}
