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
            min: 1e-10,
            max: 1e-4,
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
    pub fn forward(&self, pairs: Tensor<B, 2, Int>) -> Tensor<B, 2, Float> {
        let dims = pairs.dims();

        /*

        // Calculate the Poincaré distance
        let x = self
            .embedding_token
            .forward(pairs.clone().slice([0..dims[0], 0..1]));
        let y = self
            .embedding_token
            .forward(pairs.slice([0..dims[0], 1..2]));

        // let y = self.layer_norm.forward(y);
        // let x = self.layer_norm.forward(x);

        poincare_distance(x, y)
        */

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
        distance.squeeze(1)
    }

    pub fn forward_regression(
        &self,
        pairs: Tensor<B, 2, Int>,
        distances: Tensor<B, 2, Float>,
    ) -> RegressionOutput<B> {
        let predicted_distances = self.forward(pairs);

        //if predicted_distances.clone().to_data().convert::<f32>().value.iter().any(|x| x.is_nan()) {
        //println!("NaN detected in distances");
        //}

        //if predicted_distances.clone().to_data().convert::<f32>().value.iter().any(|x| x.is_infinite()) {
        //println!("Infinity detected in distances");
        //}

        let loss = (predicted_distances.clone() - distances.clone())
            .powf_scalar(2.0)
            .mean();

        let output = RegressionOutput::new(loss, predicted_distances, distances);
        output
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

    fn optimize<AB, O>(self, optim: &mut O, lr: f64, grads: GradientsParams) -> Self
    where
        AB: AutodiffBackend,
        O: Optimizer<Self, AB>,
        Self: AutodiffModule<AB>,
    {
        /*
        self.embedding_token.weight = self.embedding_token.weight.map(|x|
            x.clamp(0.0, 1.0)
        );
        */

        // todo move to after optimize step
        // self.embedding_token.clone().weight.map(|x| {
            //normalize_to_poincare_ball(x)
        //});

        optim.step(lr, self, grads)
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
    // let device = x.device();

    let x_norm = l2_norm(x.clone()); //.clamp(1e-5, 1.0 - 1e-5);
    let y_norm = l2_norm(y.clone()); // .clamp(1e-5, 1.0 - 1e-5);

    let diff = x - y;
    let diff_norm = l2_norm(diff).powf_scalar(2.0);

    let num = diff_norm * 2.0;
    // let num = num.add_scalar(1e-8);
    let one = Tensor::<B, 2>::ones_like(&x_norm);
    let denom =
        (one.clone() - x_norm.clone().powf_scalar(2.0)) * (one - y_norm.clone().powf_scalar(2.0));
    // let denom = denom.add_scalar(1e-8);

    let distance = num / denom;

    // 1 + relu distance
    // let ones_like = Tensor::<B, 2>::ones_like(&distance);
    // let distance = ones_like + burn::tensor::activation::relu(distance);

    acosh(distance)
}

pub fn l2_norm<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 2> {
    x.powf_scalar(2.0).sum_dim(2).sqrt().squeeze(2)
}

fn acosh<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    // let x = x.clamp_min(1.000005);
    assert!(
        x.clone()
            .greater_equal(Tensor::<B, 2>::ones_like(&x))
            .all()
            .into_scalar(),
        "acosh(x) input must be >= 1"
    );
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
    #[config(default = 16)]
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
    use burn_wgpu::{AutoGraphicsApi, Wgpu, OpenGl};
    use burn::backend::autodiff::Autodiff;
    use burn::tensor::Tensor;

    #[test]
    fn test_poincare_distance() {
        type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
        let device = burn_wgpu::WgpuDevice::default();

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

        println!("{}", distance);
        println!("{}", expected);

        let equals = distance.equal(expected).all();
        let equals = equals.into_data().value[0];

        assert!(equals);
    }
}
