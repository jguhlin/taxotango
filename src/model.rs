use burn::backend::autodiff::grads::Gradients;
use burn::lr_scheduler;
use burn::lr_scheduler::LrScheduler;
use burn::record::Recorder;
use burn::train::metric::LearningRateMetric;
use burn::{
    data::dataloader::{batcher::Batcher, DataLoaderBuilder},
    module::AutodiffModule,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig},
    optim::{
        adaptor::OptimizerAdaptor, AdamConfig, AdamWConfig, GradientsParams, Optimizer, SgdConfig,
        SimpleOptimizer,
    },
    prelude::*,
    record::CompactRecorder,
    record::Record,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{CpuTemperature, LossMetric},
        LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
    LearningRate,
};

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::decay::{WeightDecay, WeightDecayConfig};
use burn::optim::momentum::{Momentum, MomentumConfig, MomentumState};

use nn::loss::MseLoss;
use nn::{Linear, LinearConfig};
use rerun::{demo_util::grid, external::glam};
use serde::{Deserialize, Serialize};

use crate::L2Norm;
use crate::PoincareDistance;

#[derive(Config)]
pub struct LrWarmUpLinearDecaySchedulerConfig {
    initial_lr: LearningRate,
    // The final learning rate.
    top_lr: LearningRate,
    // The number of iterations before reaching the top learning rate.
    num_iters: usize,
    // The number of iterations to decay the learning rate.
    decay_iters: usize,
    // The minimum learning rate.
    min_lr: LearningRate,
}

impl LrWarmUpLinearDecaySchedulerConfig {
    pub fn init(&self) -> LrWarmUpLinearDecayScheduler {
        LrWarmUpLinearDecayScheduler {
            initial_lr: self.initial_lr,
            top_lr: self.top_lr,
            num_iters: self.num_iters,
            decay_iters: self.decay_iters,
            min_lr: self.min_lr,
            current_iter: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LrWarmUpLinearDecayScheduler {
    initial_lr: LearningRate,
    top_lr: LearningRate,
    num_iters: usize,
    decay_iters: usize,
    min_lr: LearningRate,
    current_iter: usize,
}

impl<B: Backend> LrScheduler<B> for LrWarmUpLinearDecayScheduler {
    type Record = (LearningRate, f64, usize);

    fn step(&mut self) -> LearningRate {
        self.current_iter += 1;

        if self.current_iter < self.num_iters {
            let alpha = self.current_iter as f64 / self.num_iters as f64;
            let lr = self.initial_lr * (1.0 - alpha) + self.top_lr * alpha;
            lr
        } else if self.current_iter < self.num_iters + self.decay_iters {
            let alpha = (self.current_iter - self.num_iters) as f64 / self.decay_iters as f64;
            let lr = self.top_lr * (1.0 - alpha) + self.min_lr * alpha;
            lr
        } else {
            self.min_lr
        }
    }

    fn to_record(&self) -> Self::Record {
        (self.initial_lr, self.top_lr, self.current_iter)
    }

    fn load_record(mut self, record: Self::Record) -> Self {
        self.initial_lr = record.0;
        self
    }
}

#[derive(Config)]
pub struct RiemannianSgdConfig {
    /// [Weight decay](WeightDecayConfig) config.
    weight_decay: Option<WeightDecayConfig>,
    /// [Momentum](MomentumConfig) config.
    momentum: Option<MomentumConfig>,
    /// [Gradient Clipping](GradientClippingConfig) config.
    gradient_clipping: Option<GradientClippingConfig>,
}

#[derive(Clone)]
pub struct RiemannianSgd<B: Backend> {
    momentum: Option<Momentum<B>>,
    weight_decay: Option<WeightDecay<B>>,
}

/// State of [RiemannianSgd](RiemannianSgd).
#[derive(Record, Clone)]
pub struct RiemannianSgdState<B: Backend, const D: usize> {
    momentum: Option<MomentumState<B, D>>,
}

impl<B: Backend, const D: usize> RiemannianSgdState<B, D> {
    pub fn new(momentum: Option<MomentumState<B, D>>) -> Self {
        Self { momentum }
    }
}

impl RiemannianSgdConfig {
    /// Creates a new [RiemannianSgdConfig](RiemannianSgdConfig) with default values.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<RiemannianSgd<B::InnerBackend>, M, B> {
        let momentum = self.momentum.as_ref().map(Momentum::new);
        let weight_decay = self.weight_decay.as_ref().map(WeightDecay::new);

        println!("Weight decay? {}", weight_decay.is_some());
        println!("Momentum? {}", momentum.is_some());
        println!("Grad Clipping? {}", self.gradient_clipping.is_some());

        let mut optim = OptimizerAdaptor::from(RiemannianSgd {
            momentum,
            weight_decay,
        });

        if let Some(config) = &self.gradient_clipping {
            optim = optim.with_grad_clipping(config.init());
        }

        optim
    }
}

impl<B: Backend> RiemannianSgd<B> {
    fn mobius_add<const D: usize>(&self, x: Tensor<B, D>, y: Tensor<B, D>) -> Tensor<B, D> {
        let x2 = x.clone().powf_scalar(2.0).sum_dim(D - 1);
        let y2 = y.clone().powf_scalar(2.0).sum_dim(D - 1);
        let xy = (x.clone() * y.clone()).sum_dim(D - 1);

        let ones = Tensor::<B, D>::ones_like(&x2);

        let num = ((xy.clone().mul_scalar(2.0).add_scalar(1.0) + y2.clone()) * x)
            + ((ones - x2.clone()) * y);
        let denom = xy.mul_scalar(2.0).add_scalar(1.0) + (x2 * y2);

        num / denom.clamp_min(1e-15)
    }

    fn expm<const D: usize>(&self, p: Tensor<B, D>, u: Tensor<B, D>) -> Tensor<B, D> {
        // Calculate the norm of u
        let norm = u
            .clone()
            .powf_scalar(2.0)
            .sum_dim(D - 1)
            .sqrt()
            .clamp_min(1e-10);

        // Calculate lambda_x(p), which is a scaling factor based on the point p
        let p_sqnorm = p.clone().powf_scalar(2.0).sum_dim(D - 1);
        let ones = Tensor::<B, D>::ones_like(&p_sqnorm);
        let twos = Tensor::<B, D>::full(p_sqnorm.shape(), 2.0, &p.device());
        let lambda_x = twos / (ones.sub(p_sqnorm)).clamp(1e-15, f64::INFINITY);

        // Scale u by tanh(0.5 * lambda_x(p) * norm) / norm
        let scaled_u = (lambda_x.mul_scalar(0.5) * norm.clone()).tanh() * u / norm.clamp_min(1e-15);

        // Perform the Möbius addition
        self.mobius_add(p, scaled_u)
    }

    // Custom gradient scaling for the Riemannian manifold
    fn grad<const D: usize>(&self, p: Tensor<B, D>, grad: Tensor<B, D>) -> Tensor<B, D> {
        // let p_sqnorm = p.powf_scalar(2.0).sum_dim(D - 1);
        // let ones = Tensor::<B, D>::ones_like(&p_sqnorm);
        // grad * ((ones - p_sqnorm).powf_scalar(2.0).div_scalar(4.0))

        let p_sqnorm = p.powf_scalar(2.0).sum_dim(D - 1);
        let scaling = ((Tensor::<B, D>::ones_like(&p_sqnorm).sub(p_sqnorm))
            .powf_scalar(2.0)
            .div_scalar(4.0))
        .clamp_min(1e-12);
        grad * scaling
    }

    fn project_to_manifold<const D: usize>(&self, tensor: Tensor<B, D>) -> Tensor<B, D> {
        // Calculate the L2 norm of the tensor
        let squared = tensor.clone().powf_scalar(2.0);
        let sum_squares = squared.sum_dim(D - 1); // Sum across the last dimension
        let norm = sum_squares.sqrt().clamp(1e-10, 1.0 - 1e-10); // Compute the norm and clamp

        // Normalize the tensor, ensuring it's within the manifold
        let scaled_tensor = tensor.clone() / norm.clone();

        // Optionally: Only project if norm is significantly different from 1.0
        let greater_than = Tensor::full([1], 1.0 - 1e-3, &tensor.device());
        let should_project = norm.clone().max().greater(greater_than).into_scalar();
        if should_project {
            scaled_tensor
        } else {
            tensor
        }
    }
}
impl<B: Backend> SimpleOptimizer<B> for RiemannianSgd<B> {
    type State<const D: usize> = RiemannianSgdState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        mut grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let mut state_momentum = None;

        if let Some(state) = state {
            state_momentum = state.momentum;
        }

        if let Some(weight_decay) = &self.weight_decay {
            grad = weight_decay.transform(grad, tensor.clone());
        }

        // Apply the custom Riemannian gradient scaling
        grad = self.grad(tensor.clone(), grad);

        if let Some(momentum) = &self.momentum {
            let (grad_out, state) = momentum.transform(grad, state_momentum);
            state_momentum = Some(state);
            grad = grad_out;
        }

        let state = RiemannianSgdState::new(state_momentum);

        let delta = grad.mul_scalar(-lr);

        // Update parameters using the exponential map
        let updated_tensor = self.expm(tensor, delta);

        let projected_tensor = self.project_to_manifold(updated_tensor);

        (projected_tensor, Some(state))
        // (updated_tensor, Some(state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &B::Device) -> Self::State<D> {
        state.momentum = state.momentum.map(|state| state.to_device(device));
        state
    }
}

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
    pub embedding_token: Embedding<B>,
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
            min: -0.005,
            max: 0.005,
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
        // let distances: Tensor<B, 3> = distances.unsqueeze_dims(&[-1]);
        distances
        // self.scaling_layer.forward(distances).squeeze(2)

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
            .map(|tensor| tensor.reshape([1, N]))
            .collect();

        let distances = items
            .iter()
            .map(|item| TensorData::from([item.distances]))
            .map(|data| Tensor::<B, 2>::from_data(data.convert::<u32>(), &self.device))
            .map(|tensor| tensor.reshape([1, N]))
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

// Training stuff
#[derive(Config)]
pub struct TrainingConfig {
    pub model: PoincareTaxonomyEmbeddingModelConfig,
    // pub optimizer: AdamConfig,
    // pub optimizer: SgdConfig,
    pub optimizer: AdamWConfig,
    // pub optimizer: RiemannianSgdConfig,
    #[config(default = 2048)]
    pub num_epochs: usize,
    #[config(default = 1024)]
    pub batch_size: usize,
    #[config(default = 1)]
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

    let lr_schedule = LrWarmUpLinearDecaySchedulerConfig {
        initial_lr: 1e-10,
        top_lr: 2e-4,
        num_iters: 20_000, // 100_000 is better, but for testing...
        decay_iters: 1_000_000,
        min_lr: 1e-6,
    };

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
            lr_schedule.init(),
            // 1e-5,
            // burn::lr_scheduler::linear::LinearLrSchedulerConfig::new(4e-3, 1e-6, 10_000).init(),
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

    let optim = AdamWConfig::new();

    let config = PoincareTaxonomyEmbeddingModelConfig {
        taxonomy_size: batch_gen.taxonomy_size(),
        embedding_size: 2,
    };

    B::seed(1337);

    let lr_schedule = LrWarmUpLinearDecaySchedulerConfig {
        initial_lr: 1e-10,
        top_lr: 2e-4,
        num_iters: 20_000, // 100_000 is better, but for testing...
        decay_iters: 1_000_000,
        min_lr: 1e-6,
    };

    let config = TrainingConfig::new(config, optim.clone());

    // Create the model and optimizer.
    let mut model: PoincareTaxonomyEmbeddingModel<B> = config.model.init(device);
    let mut optim = optim.init();

    let batcher_train: TangoBatcher<B> = TangoBatcher::<B>::new(device.clone());
    let batcher_valid = TangoBatcher::<B::InnerBackend>::new(device.clone());

    let taxa_levels_in_order = batch_gen.levels_in_order.clone();
    let taxa_names = batch_gen.taxa_names.clone();

    // Combine taxa level and taxa name
    let per_node_string: Vec<String> = taxa_levels_in_order
        .iter()
        .zip(taxa_names.iter())
        .map(|(level, name)| format!("{}: {}", level, name))
        .collect();

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

    let mut lr = lr_schedule.init();

    // Iterate over our training and validation loop for X epochs.
    for epoch in 1..config.num_epochs + 1 {
        /*
        let embedding_weights = model.embedding_token.weight.val().into_data();
        let j = embedding_weights.to_vec::<f32>().unwrap();

        // Chunks into dimensions (here, 3)
        let mut chunks = j.chunks(3);

        rec.log(
            "points",
            &rerun::Points3D::new(
                chunks
                    .by_ref()
                    .map(|chunk| glam::Vec3::new(chunk[0], chunk[1], chunk[2])),
            ).with_colors(colors.clone())
            .with_labels(per_node_string.clone()),
        ).expect("Failed to log points"); */

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

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);

            model = optim.step(<LrWarmUpLinearDecayScheduler as LrScheduler<B>>::step(&mut lr), model, grads);
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

pub fn inference<B: Backend>(artifact_dir: &str, device: B::Device, item: TaxaDistance<1>) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);

    let batcher = TangoBatcher::<B>::new(device.clone());
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.origins, batch.branches);

    println!("{}", model.scaling_layer.weight.val());

    println!("Inference");
    println!("Predicted {} Expected {}", output, batch.distances);
}
