use burn::lr_scheduler::LrScheduler;
use burn::module::AutodiffModule;
use burn::optim::SgdConfig;
use burn::record::Recorder;
use burn::train::metric::LearningRateMetric;
use burn::{
    data::dataloader::{batcher::Batcher, DataLoaderBuilder},
    nn::loss::MseLoss,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{metric::LossMetric, LearnerBuilder, TrainOutput, TrainStep, ValidStep},
    LearningRate,
};
// use rerun::external::glam;

use crate::*;

use std::io::Write;

// Training stuff
#[derive(Config)]
pub struct TrainingConfig {
    pub model: PoincareEmbeddingModelConfig,
    // pub optimizer: AdamConfig,
    // pub optimizer: SgdConfig,
    // pub optimizer: AdamWConfig,
    // pub optimizer: RiemannianSgdConfig,
    // pub optimizer: RiemannianAMSGradConfig,
    pub optimizer: RiemannianAdamWConfig,
    #[config(default = 16384)]
    pub num_epochs: usize,
    // #[config(default = 4)]
    // #[config(default = 8)]
    // #[config(default = 16)]
    // #[config(default = 1024)]
    // #[config(default = 4096)]
    // #[config(default = 8192)]
    #[config(default = 16384)]
    // #[config(default = 131072)]
    pub batch_size: usize,
    #[config(default = 1)]
    pub num_workers: usize,
    #[config(default = 1337002)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

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

impl LrScheduler for LrWarmUpLinearDecayScheduler {
    type Record<B: Backend> = (LearningRate, f64, usize);

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

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        (self.initial_lr, self.top_lr, self.current_iter)
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
        self.initial_lr = record.0;
        self
    }
}

#[derive(Clone, Debug)]
pub struct TaxaDistance<const P: usize, const N: usize> {
    pub origin: u32,
    pub nearby: [u32; P],
    pub distant: [u32; N],
    pub origin_weight_factor: f32,
    pub nearby_weight_factor: [f32; P],
    pub distant_weight_factor: [f32; N],
}

#[derive(Clone)]
pub struct TangoBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> TangoBatcher<B> {
    pub fn new(devices: Vec<B::Device>) -> Self {
        let device = devices[0].clone();
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct TangoBatch<B: Backend> {
    pub origins: Tensor<B, 2, Int>,
    pub nearby: Tensor<B, 2, Int>,
    pub distant: Tensor<B, 2, Int>,
    pub origin_weight_factor: Tensor<B, 2, Float>,
    pub nearby_weight_factor: Tensor<B, 2, Float>,
    pub distant_weight_factor: Tensor<B, 2, Float>,
}

impl<B: Backend, const P: usize, const N: usize> Batcher<TaxaDistance<P, N>, TangoBatch<B>> for TangoBatcher<B> {
    fn batch(&self, items: Vec<TaxaDistance<P, N>>) -> TangoBatch<B> {
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
            .map(|tensor| tensor.reshape([1, P]))
            .collect();

        let distant = items
            .iter()
            .map(|item| TensorData::from(item.distant))
            .map(|data| Tensor::<B, 1, Int>::from_data(data.convert::<u32>(), &self.device))
            .map(|tensor| tensor.reshape([1, N]))
            .collect();

        let origin_weight_factor = items
            .iter()
            .map(|item| TensorData::from([item.origin_weight_factor]))
            .map(|data| Tensor::<B, 1, Float>::from_data(data.convert::<f32>(), &self.device))
            .map(|tensor| tensor.reshape([1, 1]))
            .collect();

        let nearby_weight_factor = items
            .iter()
            .map(|item| TensorData::from(item.nearby_weight_factor))
            .map(|data| Tensor::<B, 1, Float>::from_data(data.convert::<f64>(), &self.device))
            .map(|tensor| tensor.reshape([1, P]))
            .collect();

        let distant_weight_factor = items
            .iter()
            .map(|item| TensorData::from(item.distant_weight_factor))
            .map(|data| Tensor::<B, 1, Float>::from_data(data.convert::<f64>(), &self.device))
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
        let origin_weight_factor = Tensor::cat(origin_weight_factor, 0).to_device(&self.device);
        let nearby_weight_factor = Tensor::cat(nearby_weight_factor, 0).to_device(&self.device);
        let distant_weight_factor = Tensor::cat(distant_weight_factor, 0).to_device(&self.device);

        TangoBatch {
            origins,
            nearby,
            distant,
            origin_weight_factor,
            nearby_weight_factor,
            distant_weight_factor,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<TangoBatch<B>, EmbeddingOutput<B>>
    for PoincareEmbeddingModel<B>
{
    fn step(&self, batch: TangoBatch<B>) -> TrainOutput<EmbeddingOutput<B>> {
        let item = self.forward_distance(batch.origins, batch.nearby, batch.distant, batch.origin_weight_factor, batch.nearby_weight_factor, batch.distant_weight_factor);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<TangoBatch<B>, EmbeddingOutput<B>> for PoincareEmbeddingModel<B> {
    fn step(&self, batch: TangoBatch<B>) -> EmbeddingOutput<B> {
        self.forward_distance(batch.origins, batch.nearby, batch.distant, batch.origin_weight_factor, batch.nearby_weight_factor, batch.distant_weight_factor)
    }
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<const P: usize, const N: usize, B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    mut batch_gen: crate::BatchGenerator<P, N>,
    devices: Vec<B::Device>,
) {

    let device = &devices[0];

    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    
        let lr_schedule = LrWarmUpLinearDecaySchedulerConfig {
            initial_lr: 5e-4,     // Start with a small but reasonable initial LR
            // top_lr: 25e-4,         // Peak LR during training
            // top_lr: 1e-3,
            top_lr: 6e-2,
            num_iters: 100,    // Warm-up over 70,000 iterations (~500 epochs - on total dataset)
            decay_iters: 2_000_000, // Total training iterations
            min_lr: 1e-7,         // Minimum LR to decay to
        };

    B::seed(config.seed);

    let batcher_train: TangoBatcher<B> = TangoBatcher::<B>::new(devices.clone());
    let batcher_valid = TangoBatcher::<B::InnerBackend>::new(devices.clone());

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
            config.model.init::<B>(vec![Default::default()]),
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

pub fn custom_training_loop<const P: usize, const N: usize, B: AutodiffBackend>(
    batch_gen: crate::BatchGenerator<P, N>,
    devices: Vec<B::Device>,
) {
    println!("Starting training loop");

    let device = &devices[0];

    let dims = 2;

    let lr_schedule = LrWarmUpLinearDecaySchedulerConfig {
        initial_lr: 1e-5,     // Start with a small but reasonable initial LR
        // top_lr: 25e-4,         // Peak LR during training
        // top_lr: 1e-3,
        top_lr: 5e-4,
        num_iters: 1000*10,    // Warm-up over 70,000 iterations (~500 epochs - on total dataset) // * 10 for super small batch sizes
        decay_iters: 1_000_000,
        // decay_iters: 2_000_000, // Total training iterations
        min_lr: 1e-7,         // Minimum LR to decay to
    };


    /*
    let rec = rerun::RecordingStreamBuilder::new("rerun_poincare")
        .connect()
        .expect("Failed to start recording stream");
    */

    // let optim = AdamWConfig::new();
    //let optim = RiemannianSgdConfig::new()
        //.with_gradient_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(1.0)));
    // let optim = SgdConfig::new();
    // let optim = RiemannianAMSGradConfig::new();
    let optim = RiemannianAdamWConfig::new();
        // .with_gradient_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(1.0)))
        // .with_weight_decay(1e-5);

    let mut loss_history_fh = std::fs::File::create("loss_history.txt").unwrap();
    let mut loss_history_writer = std::io::BufWriter::new(&mut loss_history_fh);

    // let names = taxa_dist.branches.iter().map(|x| graph.raw_nodes()[*x as usize].weight.name.clone()).collect::<Vec<_>>();
    /*
    let names = batch_gen
        .graph
        .raw_nodes()
        .iter()
        .map(|x| x.weight.name.clone())
        .collect::<Vec<_>>();

    let taxa_levels = batch_gen
        .graph
        .raw_nodes()
        .iter()
        .map(|x| x.weight.rank_str.clone())
        .collect::<Vec<_>>();

    // Labels (taxa | name)
    let taxa_levels = taxa_levels
        .iter()
        .zip(names.iter())
        .map(|(a, b)| format!("{} | {}", a, b))
        .collect::<Vec<_>>();

    let colors = batch_gen
        .graph
        .raw_nodes()
        .iter()
        .map(|x| x.weight.color)
        .collect::<Vec<_>>(); */

    let config = PoincareEmbeddingModelConfig {
        taxonomy_size: batch_gen.taxonomy_size(),
        embedding_size: dims, // 3,
        root_idx: batch_gen.root.index(),
    };

    B::seed(1337);

    // let base_lr = 8e-3;
    // let mut lr = base_lr;

    let config = TrainingConfig::new(config, optim.clone());

    // Create the model and optimizer.
    let mut model: PoincareEmbeddingModel<B> = config.model.init(devices.clone());
    // let td: Tensor<B, 2, Float> = Tensor::from_data(batch_gen.sample_weights.clone().as_slice(), device);
    // model.sample_weights = td;

    let mut optim = optim.init();

    let batcher_train: TangoBatcher<B> = TangoBatcher::<B>::new(devices.clone());
    // let batcher_valid = TangoBatcher::<B::InnerBackend>::new(device.clone());

    // let ds_valid = batch_gen.valid();

    let validation = Arc::clone(&batch_gen.graph);
    let batch_gen = Arc::new(batch_gen);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(Arc::clone(&batch_gen));

    /*let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ds_valid);
    */

    let mut lr = lr_schedule.init();

    let mut total_iter = 0;

    let mut rng = rand::thread_rng();
    let mut all_mares = Vec::new();

    let mut lr_actual: f64 = 0.9;

    // Iterate over our training and validation loop for X epochs.
    for epoch in 1..config.num_epochs + 1 {
        println!("Starting epoch {}", epoch);
        model.epoch = epoch as u64;

        if epoch == 1000 {
            batch_gen
                .weighted_sampling
                .store(false, std::sync::atomic::Ordering::Relaxed);

            model.warmup = false;
        }

        // These values for RAdamW
        /*
        if epoch <= 100_000 {
            lr_actual = 8e-5;
        }

        if epoch <= 8000 {
            lr_actual = 2e-4;
        }

        if epoch <= 4000 {
            lr_actual = 4e-4;
        } 

        if epoch <= 2000 {
            lr_actual = 6e-4;
        } 

        if epoch <= 1000 {
            lr_actual = 8e-4;
        }

        if epoch <= 500 {
            lr_actual = 4e-3;
        }
         */

        lr_actual = 8e-3;

        if epoch <= 150 {
            lr_actual = 8e-4;
        }

        if epoch >= 1200 {
            lr_actual = 6e-5;
        }

        if epoch >= 4000 {
            lr_actual = 4e-5;
        }

        if epoch >= 8000 {
            lr_actual = 2e-5;
        }

        if epoch >= 10_000 {
            lr_actual = 8e-6
        }
        

        /*
       // For Riemannian SGD
        if epoch <= 50 {
            lr_actual = 0.3;
        } else if epoch <= 100 {
            lr_actual = 0.6;
        } else if epoch > 200 {
            lr_actual = 0.45;
        } else if epoch > 500 {
            lr_actual = 0.25;
        } */

        /*
        if epoch <= 10000 {
            lr = 3e-6;
        }

        if epoch <= 4000 {
            lr = 3e-5;
        }

        if epoch <= 2000 {
            lr = 3e-4;
        }

        if epoch <= 1000 {
            lr = 3e-3;
        }

        if epoch <= 500 {
            lr = 2e-4;
        }
        */

        let artifact_dir = "/mnt/data/data/poincare_embeddings";

        // Checkpoint every 10 epochs
        if epoch % 10 == 0 {
            model
                .clone()
                .save_file(
                    format!("{artifact_dir}/model_{}", epoch),
                    &CompactRecorder::new(),
                )
                .expect("Trained model should be saved successfully");
        }

        let mut avg_loss = 0.0;

        println!("Starting dataloader_train");
        // Implement our training loop.
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            // let output = model.forward(batch.origins);
            // let nearby = model.forward(batch.nearby);
            // let distant = model.forward(batch.distant);

            // let poincare_loss = PoincareLoss::new().forward(output, nearby, distant, burn::nn::loss::Reduction::Mean);

            // let lr_actual = <LrWarmUpLinearDecayScheduler as LrScheduler<B>>::step(&mut lr);

            let poincare_loss = model.forward_distance(batch.origins, batch.nearby, batch.distant, batch.origin_weight_factor, batch.nearby_weight_factor, batch.distant_weight_factor);

            /*if avg_loss == 0.0 {
                avg_loss = poincare_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            } else {
                avg_loss = (avg_loss + poincare_loss.clone().into_data().to_vec::<f32>().unwrap()[0])/2.0;
            }*/

            if avg_loss == 0.0 {
                avg_loss = poincare_loss
                    .loss
                    .clone()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()[0];
            } else {
                avg_loss = (avg_loss
                    + poincare_loss
                        .loss
                        .clone()
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap()[0])
                    / 2.0;
            }

            // lr_actual = <LrWarmUpLinearDecayScheduler as LrScheduler<B>>::step(&mut lr);

            if iteration == 0 { // || iteration % 10 == 0 {

                // Rerun not compatible with burn 0.15
                /*

                let embedding_weights = model.embedding_token.weight.val().into_data();
                let j = embedding_weights.to_vec::<f32>().unwrap();

                // Chunks into dimensions
                let mut chunks = j.chunks(dims);

                // todo have to put thru a PCA or something

                if dims >= 3 {
                    if dims > 3 {
                        println!("More than 3 dimensions, truncating to 3");
                    }
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
                    )
                    .expect("Failed to log points");
                } else if dims == 2 {
                    rec.log(
                        "points",
                        &rerun::Points2D::new(
                            chunks
                                .by_ref()
                                .map(|chunk| glam::Vec2::new(chunk[0], chunk[1])),
                        )
                        .with_colors(colors.clone())
                        .with_labels(taxa_levels.clone()),
                        //.with_colors(colors.clone())
                        // .with_labels(per_node_string.clone()),
                    )
                    .expect("Failed to log points");
                }
                 */

                println!(
                    "[Train - Epoch {} - Iteration {}/{} - Lr {:.8}] Loss {:.6}",
                    epoch, iteration, total_iter, lr_actual, avg_loss,
                );

                avg_loss = 0.0;
            }

            let grads: <B as AutodiffBackend>::Gradients = poincare_loss.loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);

            model = optim.step(lr_actual, model, grads);

            // Put root node back at the origin

            model.embedding_token.weight = model.embedding_token.weight.map(|x| {
                let mut x = Tensor::inner(x);
                let norms = l2_norm(x.clone()); // [total_embeddings, 1]

                let retracted = x.clone() / norms.clone().add_scalar(1e-6);

                let norms: Tensor<<B as AutodiffBackend>::InnerBackend, 1> = norms.squeeze(1); // [total_embeddings]

                let gte: Tensor<<B as AutodiffBackend>::InnerBackend, 1, Bool> = norms.clone().greater_equal_elem(1.0); // [total_embeddings]

                /*
                if gte.clone().any().into_scalar() {

                    // println!("Found norms greater than 1.0 - Retracting");
                    // println!("Norms found: {}", norms.clone().select(0, norms.clone().greater_equal_elem(1.0).argwhere().squeeze(1))); // [total_embeddings]

                    // Print out the embeddings that are greater than 1.0
                    // let too_big = x.clone().select(0, norms.clone().greater_equal_elem(1.0).argwhere().squeeze(1));
                    // println!("Too big: {}", too_big);
                    // println!("Replacing with retracted: {}", retracted.clone().select(0, norms.clone().greater_equal_elem(1.0).argwhere().squeeze(1)));

                }  */

                let norms = norms.reshape([x.shape().dims[0], 1]);

                x = x.clone().mask_where(norms.clone().greater_equal_elem(1.0).expand(x.clone().shape()), retracted); // [total_embeddings, dims]

                let x = x.set_require_grad(true);

                //if gte.clone().any().into_scalar() {
                    //println!("Replaced: {}", x.clone().select(0, gte.clone().argwhere().squeeze(1)));
                //}

                Tensor::from_inner(x)
            });

            model.embedding_token.weight = model.embedding_token.weight.set_require_grad(true);

            // Retraction step

            // model.embedding_token.weight = model.embedding_token.weight.map(|x| retraction(x));

            let norms = l2_norm(model.embedding_token.weight.val().clone());
            let gte = norms.greater_equal_elem(1.0);
            // if gte.any().into_scalar() {
                // panic!("Norms greater than 1.0");
                // model.embedding_token = optim.optim.project_to_manifold(model.embedding_token);
            // }

            total_iter += 1;
        }

        // Get the model without autodiff.
        // let model_valid = model.valid();
        // let valid_device = model_valid.embedding_token.weight.device();

        // For validation, we'll select some random taxa and compute mean reciprocal rank back to the root.

        let graph = Arc::clone(&validation);

        // Every 100 epochs, do a validation run.

        if epoch % 100 == 0 {

            // println!("Doing validation");
            let mut mares = Vec::new();

            let mut root_idx = None;

            for _i in 0..1024 {
                let mut taxa_idx = graph.node_indices().choose(&mut rng).unwrap();
                let mut current_taxa = &graph.raw_nodes()[taxa_idx.index()].weight;

                // Find a random species.
                while current_taxa.rank != TaxaLevel::Species {
                    taxa_idx = graph.node_indices().choose(&mut rng).unwrap();
                    current_taxa = &graph.raw_nodes()[taxa_idx.index()].weight;
                }
                // println!("Found species: {}", current_taxa.name);

                let query_idx = taxa_idx;

                let mut path_to_root = Vec::new();

                // Traverse to the root node and collect the path.
                // Root node is a node with no parent.

                let mut parent_edge = graph
                    .neighbors_directed(taxa_idx, petgraph::Direction::Incoming)
                    .next();

                while let Some(parent_edge_unwrapped) = parent_edge {
                    path_to_root.push(taxa_idx);
                    taxa_idx = parent_edge_unwrapped;
                    // current_taxa = &graph.raw_nodes()[taxa_idx.index()].weight;

                    parent_edge = graph
                        .neighbors_directed(taxa_idx, petgraph::Direction::Incoming)
                        .next();
                    // println!("Added parent edge {}. Current count: {}", current_taxa.name, path_to_root.len());
                }
                path_to_root.push(taxa_idx); // Add root node
                if root_idx.is_none() {
                    root_idx = Some(taxa_idx);
                } else {
                    assert_eq!(root_idx, Some(taxa_idx));
                }
                let root_idx = taxa_idx;

                let root = TensorData::from([[root_idx.index() as u32]]);
                let root: Tensor<B, 2, Int> = Tensor::from_data(root.convert::<u32>(), device);
                let root = root.reshape([1, 1]);

                let mut ranks_to_root = Vec::new();

                for (rank, taxon) in path_to_root.iter().enumerate() {
                    // Calculate distance between nodes.
                    let query_idx = *taxon;
                    let query = TensorData::from([[query_idx.index() as u32]]);
                    let query: Tensor<B, 2, Int> = Tensor::from_data(query.convert::<u32>(), device);
                    let query = query.reshape([1, 1]);

                    let distance = model.forward_simple_distance(root.clone(), query);

                    let distance = distance.into_data().to_vec::<f32>().unwrap()[0];

                    ranks_to_root.push((distance, rank));
                }

                // log::info!("Ranks to root: {:?}", ranks_to_root);

                // Sort by distance.
                ranks_to_root.sort_by(|a, b| 
                    match b.0.partial_cmp(&a.0) {
                        Some(x) => x,
                        None => {
                            log::info!("Cannot compare {} and {}", a.0, b.0);
                            // std::cmp::Ordering::Equal
                            panic!("Cannot compare {} and {}", a.0, b.0);
                        }
                    });
                // log::debug!("Ranks to root: {:?}", ranks_to_root);
                // log::info!("Ranks to root - Sorted: {:?}", ranks_to_root);

                // Find the rank (MARE) mean absolute rank error.
                let mut mare_vals = Vec::new();

                let mut mare_debug = Vec::new();

                for (sorted_rank, (_distance, expected_rank)) in ranks_to_root.iter().enumerate() {
                    if sorted_rank == 0 {
                        continue;
                    }

                    mare_debug.push((sorted_rank, *expected_rank));

                    let mare = (sorted_rank as f32 - *expected_rank as f32).abs();
                    // If it's a NAN, print the ranks
                    if mare.is_nan() {
                        // Print sorted and expected
                        println!("Expected: {:?}", path_to_root);
                    } else {
                        mare_vals.push(mare);
                    }
                }

                // log::info!("MARE Debug: {:?}", mare_debug);

                let mare: f32 = mare_vals.iter().sum::<f32>() / mare_vals.len() as f32;

                if mare.is_nan() {
                    println!("MARE is NAN");
                    println!("Sorted: {:?}", ranks_to_root);
                    println!("Expected: {:?}", path_to_root);
                } else {
                    mares.push(mare);
                }
            }

            // Compute the final MARE.
            let final_mare: f32 = mares.iter().sum::<f32>() / mares.len() as f32;
            all_mares.push(final_mare);

            // Export loss and last MARE to file
            writeln!(
                loss_history_writer,
                "{}\t{}\t{}\t{}",
                epoch,
                avg_loss,
                all_mares.last().unwrap(),
                lr_actual
            )
            .unwrap();
            // Flush the buffer
            loss_history_writer.flush().unwrap();

            println!("Epoch {} - MARE: {}", epoch, final_mare);
        }
    }

    /*
    // Implement our validation loop.
    for (iteration, batch) in dataloader_test.iter().enumerate() {
        let output = model_valid.forward(batch.origins);
        let nearby = model_valid.forward(batch.nearby);
        let distant = model_valid.forward(batch.distant);

        let loss = PoincareLoss::new().forward(output, nearby, distant, burn::nn::loss::Reduction::Sum);
    } */
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
