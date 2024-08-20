// use mimalloc::MiMalloc;

// #[global_allocator]
// static GLOBAL: MiMalloc = MiMalloc;

use burn::backend::{autodiff::Autodiff, Wgpu};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::tensor::Tensor;

use flexi_logger::FileSpec;
use taxotangolib::*;

fn main() {
    let debug = false;

    // Highest level of logging is debug
    // Lowest level of logging is error

    // env_logger::init();
    flexi_logger::Logger::try_with_str("error")
        .expect("Invalid log level")
        .log_to_file(FileSpec::default()) // write logs to file
        .start()
        .expect("FlexiLogger initialization failed");

    let nodes_file = "/mnt/data/data/nt/taxdmp/nodes.dmp";
    let names_file = "/mnt/data/data/nt/taxdmp/names.dmp";

    let mut generator = build_taxonomy_graph_generator(nodes_file, names_file, 16);

    let config = PoincareTaxonomyEmbeddingModelConfig {
        taxonomy_size: generator.taxonomy_size(),
        embedding_size: 16,
    };

    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

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
        let adamwconfig = AdamWConfig::new()
            .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(1.0)));

        crate::model::train::<8, MyAutodiffBackend>(
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
