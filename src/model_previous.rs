use rerun::{demo_util::grid, external::glam};
use burn::lr_scheduler::LrScheduler;
use burn::module::AutodiffModule;
use burn::record::Recorder;
use burn::train::metric::LearningRateMetric;
use burn::optim::SgdConfig;
use burn::{
    data::dataloader::{batcher::Batcher, DataLoaderBuilder},
    optim::{
        AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    nn::loss::MseLoss,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::LossMetric,
        LearnerBuilder, TrainOutput, TrainStep, ValidStep,
    },
    LearningRate,
};

use crate::*;

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

#[derive(Clone, Debug)]
pub struct TaxaDistance<const N: usize> {
    pub origin: u32,
    pub nearby: [u32; 1],
    pub distant: [u32; N],
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
    pub nearby: Tensor<B, 2, Int>,
    pub distant: Tensor<B, 2, Int>,
}

impl<B: Backend, const N: usize> Batcher<TaxaDistance<N>, TangoBatch<B>> for TangoBatcher<B> {
    fn batch(&self, items: Vec<TaxaDistance<N>>) -> TangoBatch<B> {

        let origins = items
            .iter()
            .map(|item| TensorData::from([item.origin]))
            .map(|data| Tensor::<B, 1, Int>::from_data(data.convert::<u32>(), &self.device))
            .map(|tensor| tensor.reshape([1, 1]))
            .collect();

        let nearby = items
            .iter()
            .map(|item| TensorData::from(item.nearby))
            .map(|data| Tensor::<B, 1, Int>::from_data(data.convert::<u32>(), &self.device))
            .map(|tensor| tensor.reshape([1, 1]))
            .collect();

        let distant = items
            .iter()
            .map(|item| TensorData::from(item.distant))
            .map(|data| Tensor::<B, 1, Int>::from_data(data.convert::<u32>(), &self.device))
            .map(|tensor| tensor.reshape([1, N]))
            .collect();

        /*

        let distances = items
            .iter()
            .map(|item| TensorData::from([item.distances]))
            .map(|data| Tensor::<B, 2>::from_data(data.convert::<u32>(), &self.device))
            .map(|tensor| tensor.reshape([1, N]))
            .collect();
        */

        let origins = Tensor::cat(origins, 0).to_device(&self.device);
        let nearby = Tensor::cat(nearby, 0).to_device(&self.device);
        let distant = Tensor::cat(distant, 0).to_device(&self.device);

        TangoBatch {
            origins,
            nearby,
            distant
        }
    }
}

impl<B: AutodiffBackend> TrainStep<TangoBatch<B>, EmbeddingOutput<B>>
    for PoincareEmbeddingModel<B>
{
    fn step(&self, batch: TangoBatch<B>) -> TrainOutput<EmbeddingOutput<B>> {
        let item = self.forward_distance(batch.origins, batch.nearby, batch.distant);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<TangoBatch<B>, EmbeddingOutput<B>>
    for PoincareEmbeddingModel<B>
{
    fn step(&self, batch: TangoBatch<B>) -> EmbeddingOutput<B> {
        self.forward_distance(batch.origins, batch.nearby, batch.distant)
    }
}

// Training stuff
#[derive(Config)]
pub struct TrainingConfig {
    pub model: PoincareEmbeddingModelConfig,
    // pub optimizer: AdamConfig,
    // pub optimizer: SgdConfig,
    // pub optimizer: AdamWConfig,
    // pub optimizer: RiemannianSgdConfig,
    pub optimizer: RiemannianAMSGradConfig,
    #[config(default = 2048)]
    pub num_epochs: usize,
    // #[config(default = 4)]
    // #[config(default = 8192)]
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 8)]
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
        num_iters: 10_000, // 100_000 is better, but for testing...
        decay_iters: 100_000,
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

    let rec = rerun::RecordingStreamBuilder::new("rerun_poincare").connect().expect("Failed to start recording stream");

    // let optim = AdamWConfig::new();
    // let optim = RiemannianSgdConfig::new();
    // let optim = SgdConfig::new();
    let optim = RiemannianAMSGradConfig::new();

    // let names = taxa_dist.branches.iter().map(|x| graph.raw_nodes()[*x as usize].weight.name.clone()).collect::<Vec<_>>();
    let names = batch_gen.graph.raw_nodes().iter().map(|x| x.weight.name.clone()).collect::<Vec<_>>();
    
    let taxa_levels = batch_gen.graph.raw_nodes().iter().map(|x| x.weight.rank_str.clone()).collect::<Vec<_>>();

    // Labels (taxa | name)
    let taxa_levels = taxa_levels.iter().zip(names.iter()).map(|(a, b)| format!("{} | {}", a, b)).collect::<Vec<_>>();

    let colors = batch_gen.graph.raw_nodes().iter().map(|x| x.weight.color).collect::<Vec<_>>();

    let config = PoincareEmbeddingModelConfig {
        taxonomy_size: batch_gen.taxonomy_size(),
        embedding_size: 3,
    };

    B::seed(1337);

    let lr_schedule = LrWarmUpLinearDecaySchedulerConfig {
        initial_lr: 1e-12,
        top_lr: 5e-4,
        num_iters: 10_000, // 100_000 is better, but for testing...
        decay_iters: 500_000,
        min_lr: 1e-8,
    };

    let base_lr = 3e-1;
    let mut lr = base_lr;

    let config = TrainingConfig::new(config, optim.clone());

    // Create the model and optimizer.
    let mut model: PoincareEmbeddingModel<B> = config.model.init(device);
    let mut optim = optim.init();

    let batcher_train: TangoBatcher<B> = TangoBatcher::<B>::new(device.clone());
    // let batcher_valid = TangoBatcher::<B::InnerBackend>::new(device.clone());

    // let ds_valid = batch_gen.valid();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(batch_gen);

    /*let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ds_valid);
    */

    // let mut lr = lr_schedule.init();

    let mut total_iter = 0;

    // Iterate over our training and validation loop for X epochs.
    for epoch in 1..config.num_epochs + 1 {

        if epoch <= 200 {
            lr = 2e-3;
        }

        if epoch > 200 {
            lr = 2e-1;
        }

        let artifact_dir = "/mnt/data/data/poincare_embeddings";

        // Checkpoint every 10 epochs
        if epoch % 10 == 0 {
            model.clone()
                .save_file(format!("{artifact_dir}/model_{}", epoch), &CompactRecorder::new())
                .expect("Trained model should be saved successfully");
        }

        let mut avg_loss = 0.0;

        // Implement our training loop.
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward(batch.origins);
            let nearby = model.forward(batch.nearby);
            let distant = model.forward(batch.distant);

            let poincare_loss = PoincareLoss::new().forward(output, nearby, distant, burn::nn::loss::Reduction::Mean);

            // let lr_actual = <LrWarmUpLinearDecayScheduler as LrScheduler<B>>::step(&mut lr);

            if avg_loss == 0.0 {
                avg_loss = poincare_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            } else {
                avg_loss = (avg_loss + poincare_loss.clone().into_data().to_vec::<f32>().unwrap()[0])/2.0;
            }

            if (iteration == 0 || total_iter % 1000 == 0) {

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
                    )
                    .with_colors(colors.clone())
                    .with_labels(taxa_levels.clone()),
                    
                     //.with_colors(colors.clone())
                    // .with_labels(per_node_string.clone()),
                ).expect("Failed to log points");

                println!(
                    "[Train - Epoch {} - Iteration {}/{} - Lr {:.8}] Loss {:.6}",
                    epoch,
                    iteration,
                    total_iter,
                    lr,
                    avg_loss,
                );

                avg_loss = 0.0;
            }

            if poincare_loss.contains_nan().into_scalar() {
                panic!("Loss contains NaN");
            }

            let grads = poincare_loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);

            model = optim.step(lr, model, grads);

            // Retraction step

            // model.embedding_token.weight = model.embedding_token.weight.map(|x| retraction(x));

            // let norms = l2_norm(model.embedding_token.weight.val().clone());
            // let ones = Tensor::<B, 2>::ones_like(&norms);
            // let gte = norms.greater_equal(ones);
            // if gte.any().into_scalar() {
                // panic!("Norms greater than 1");
            // }

            total_iter += 1;
        }

        // Get the model without autodiff.
        let model_valid = model.valid();

        /*
        // Implement our validation loop.
        for (iteration, batch) in dataloader_test.iter().enumerate() {
            let output = model_valid.forward(batch.origins);
            let nearby = model_valid.forward(batch.nearby);
            let distant = model_valid.forward(batch.distant);

            let loss = PoincareLoss::new().forward(output, nearby, distant, burn::nn::loss::Reduction::Sum);
        } */
    }
}

/*
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
*/
