// use mimalloc::MiMalloc;

// #[global_allocator]
// static GLOBAL: MiMalloc = MiMalloc;

use burn::backend::{Wgpu, autodiff::Autodiff};
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::tensor::Tensor;
use burn::data::dataset::Dataset;

use flexi_logger::FileSpec;
use taxotangolib::*;

fn main() {
    // env_logger::init();
    flexi_logger::Logger::try_with_str("debug").expect("Invalid log level")
        .log_to_file(FileSpec::default())         // write logs to file
         .start().expect("FlexiLogger initialization failed");

    let nodes_file = "/mnt/data/data/nt/taxdmp/nodes.dmp";
    let names_file = "/mnt/data/data/nt/taxdmp/names.dmp";

    let generator = build_taxonomy_graph_generator(nodes_file, names_file);

    let config = PoincareTaxonomyEmbeddingModelConfig {
        taxonomy_size: generator.taxonomy_size(),
        embedding_size: 16,
    };

    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    let model = config.init::<MyBackend>(&device);
    // Test data
    // todo

    let adamconfig = AdamConfig::new()
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(1.0)));

    crate::model::train::<MyAutodiffBackend>(
        "/mnt/data/data/taxontango_training",
        crate::model::TrainingConfig::new(config, adamconfig),
        generator,
        device,
    );

}
