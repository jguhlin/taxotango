// use mimalloc::MiMalloc;

// #[global_allocator]
// static GLOBAL: MiMalloc = MiMalloc;

use burn_wgpu::{AutoGraphicsApi, Wgpu, OpenGl};
use burn::backend::autodiff::Autodiff;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::tensor::Tensor;

use taxotangolib::*;

fn main() {
    env_logger::init();

    let nodes_file = "/mnt/data/data/nt/taxdmp/nodes.dmp";
    let names_file = "/mnt/data/data/nt/taxdmp/names.dmp";

    // build_taxonomy_graph_limit_depth_csv_output(nodes_file, names_file, 4);

    // log::debug!("Generating first test batch");
    // let batch = batch_gen.generate_batch();
    // println!("{:?}", batch);

    let generator = build_taxonomy_graph_generator(nodes_file, names_file);

    let config = PoincareTaxonomyEmbeddingModelConfig {
        taxonomy_size: generator.taxonomy_size(),
        embedding_size: 16,
        // layer_norm_eps: 1e-8,
    };

    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn_wgpu::WgpuDevice::default();

    // println!("{:#?}", device);

    // let nn: PoincareTaxonomyEmbeddingModel<Wgpu> = config.init(&device);

    // let output = nn.forward(Tensor::<MyBackend, 2, Int>::from_data(
    //[[1, 2], [3, 4], [1, 6]],
    //&device,
    //));
    //println!("{}", output);

    //let output = nn.forward_regression(
    //Tensor::<MyBackend, 2, Int>::from_data([[1, 2], [3, 4], [1, 6]], &device),
    //Tensor::<MyBackend, 2, Float>::from_data([[1.0], [2.0], [3.0]], &device),
    //);

    //println!("{}", output.output);
    //println!("{}", output.targets);

    //println!("{}", output.loss);

    // print!("{}[2J", 27 as char);
    // println!("");

    // panic!();

    let adamconfig = AdamConfig::new()
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(1.0)));

    crate::model::train::<MyAutodiffBackend>(
        "/mnt/data/data/taxontango_training",
        crate::model::TrainingConfig::new(config, adamconfig),
        generator,
        device,
    );

    // let device = Default::default();
    // Creation of two tensors, the first with explicit values and the second one with ones, with the same shape as the first
    // let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]], &device);
    // let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);

    // Print the element-wise addition (done with the WGPU backend) of the two tensors.
    // println!("{}", tensor_1 + tensor_2);
}
