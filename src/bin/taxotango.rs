// use mimalloc::MiMalloc;

// #[global_allocator]
// static GLOBAL: MiMalloc = MiMalloc;

use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};

use taxotangolib::*;

fn main() {
    env_logger::init();

    let nodes_file = "/mnt/data/data/nt/taxdmp/nodes.dmp";
    let names_file = "/mnt/data/data/nt/taxdmp/names.dmp";

    build_taxonomy_graph_limit_depth_csv_output(nodes_file, names_file, 4);

    // log::debug!("Generating first test batch");

    // let batch = batch_gen.generate_batch();
    // println!("{:?}", batch);

    /* let config = PoincareTaxonomyEmbeddingModelConfig {
        taxonomy_size: batch_gen.nodes.len(),
        embedding_size: 16,
    }; */

    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    println!("{:#?}", device);

    // let nn: PoincareTaxonomyEmbeddingModel<Wgpu> = config.init(&device);

    // let output = nn.forward(Tensor::<MyBackend, 2, Int>::from_data(
    // [[1, 2], [3, 4], [1, 6]],
    // &device,
    // ));

    // println!("{}", output);

    /*

    crate::model::train::<MyAutodiffBackend>(
        "/tmp/guide",
        crate::model::TrainingConfig::new(config, AdamConfig::new()),
        batch_gen,
        device,
    ); */

    // let device = Default::default();
    // Creation of two tensors, the first with explicit values and the second one with ones, with the same shape as the first
    // let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]], &device);
    // let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);

    // Print the element-wise addition (done with the WGPU backend) of the two tensors.
    // println!("{}", tensor_1 + tensor_2);
}
