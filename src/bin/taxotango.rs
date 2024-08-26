// use mimalloc::MiMalloc;

// #[global_allocator]
// static GLOBAL: MiMalloc = MiMalloc;

use std::sync::Arc;

// todo fusion
// use burn_fusion::Fusion;
use burn::backend::{autodiff::Autodiff, libtorch::LibTorchDevice, LibTorch, Wgpu};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::Tensor;
use flexi_logger::FileSpec;
use petgraph::algo::astar;
use rerun::{demo_util::grid, external::glam};

use taxotangolib::*;

fn main() {
    let debug = false;
    let infer = true;
    let view = false;
    let custom = false;

    if view {
        let rec = rerun::RecordingStreamBuilder::new("rerun_embeddings")
            .spawn()
            .expect("Failed to start recording stream");

        type MyBackend = Wgpu<f32, i32>;

        let device = burn::backend::wgpu::WgpuDevice::default();

        let config =
            TrainingConfig::load(format!("/mnt/data/data/taxontango_training/config.json"))
                .expect("Config should exist for the model");
        let record = CompactRecorder::new()
            .load(
                format!("/mnt/data/data/taxontango_training/model").into(),
                &device,
            )
            .expect("Trained model should exist");

        let model = config.model.init::<MyBackend>(&device).load_record(record);

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
            ),
        )
        .expect("Failed to log points");

        return;
    }

    if infer {
        let nodes_file = "/mnt/data/data/nt/taxdmp/nodes.dmp";
        let names_file = "/mnt/data/data/nt/taxdmp/names.dmp";

        let mut generator: BatchGenerator<1> =
            build_taxonomy_graph_generator(nodes_file, names_file, 1);

        // Distance from Root(1) to Peripitdae(27564)
        let root_idx = generator.nodes.get(&1).unwrap();
        let peripitdae_idx = generator.nodes.get(&27564).unwrap();

        let distance = astar(
            Arc::as_ref(&generator.graph),
            *root_idx,
            |node| node == *peripitdae_idx,
            |_| 1,
            |_| 0,
        )
        .unwrap()
        .0;

        let taxadistance = TaxaDistance {
            origin: root_idx.index() as u32,
            branches: [peripitdae_idx.index() as u32],
            distances: [distance],
        };

        type MyBackend = Wgpu<f32, i32>;
        type MyAutodiffBackend = Autodiff<MyBackend>;

        let device = burn::backend::wgpu::WgpuDevice::default();
        inference::<MyBackend>("/mnt/data/data/taxontango_training", device, taxadistance);

        generator.shutdown();
        return;
    }

    // Highest level of logging is debug
    // Lowest level of logging is error

    // env_logger::init()
    // try_with_str is the level
    flexi_logger::Logger::try_with_str("debug")
        .expect("Invalid log level")
        .log_to_file(FileSpec::default())
        .start()
        .expect("FlexiLogger initialization failed");

    log::info!("Started");

    let nodes_file = "/mnt/data/data/nt/taxdmp/nodes.dmp";
    let names_file = "/mnt/data/data/nt/taxdmp/names.dmp";

    let mut generator = build_taxonomy_graph_generator(nodes_file, names_file, 52); 

    let config = PoincareTaxonomyEmbeddingModelConfig {
        taxonomy_size: generator.taxonomy_size(),
        embedding_size: 8,
    };

    // type MyBackend = Wgpu<f32, i32>;

    tch::maybe_init_cuda();
    type MyBackend = LibTorch<f32, i8>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    // let device = burn::backend::wgpu::WgpuDevice::default();
    let device = LibTorchDevice::Cuda(0);

    // Use custom training loop
    if custom {
        custom_training_loop::<4, MyAutodiffBackend>(generator, &device);

        return;
    }

    if debug {
        let model = config.init::<MyBackend>(&device);
        // let batch = TaxaDistance {
        //origin: 1,
        //branches: [1, 2, 3, 4, 5, 6, 7, 8],
        //distances: [1, 2, 3, 4, 5, 6, 7, 8],
        //};

        let tb = TangoBatcher::new(device);
        let batch = tb.batch(
            (0..16)
                .map(|i| generator.get(i).unwrap())
                .collect::<Vec<_>>(),
        );

        println!("{:#?}", batch);
        println!("{:#?}", batch.origins);

        let output = model.forward(batch.origins, batch.branches);
        println!("{}", output);
        generator.shutdown();
    } else {
        let adamwconfig = AdamWConfig::new();
        // .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(1.0)));

        crate::model::train::<4, MyAutodiffBackend>(
            "/mnt/data/data/taxontango_training",
            crate::model::TrainingConfig::new(config, adamwconfig),
            generator,
            device,
        );

        // crate::model::custom_training_loop::<MyAutodiffBackend>(generator, &device);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batcher() {
        let generator = BatchGenerator::testing();

        let config = PoincareTaxonomyEmbeddingModelConfig {
            taxonomy_size: generator.taxonomy_size(),
            embedding_size: 3,
        };

        type MyBackend = Wgpu<f32, i32>;

        let device = burn::backend::wgpu::WgpuDevice::default();

        let model = config.init::<MyBackend>(&device);

        let tb = TangoBatcher::new(device);
        let batch = tb.batch(
            (0..10)
                .map(|i| generator.get(i).unwrap())
                .collect::<Vec<_>>(),
        );

        let output = model.forward(batch.origins, batch.branches);
        // println!("{:#?}", output);
        println!("{}", output);
    }
}
