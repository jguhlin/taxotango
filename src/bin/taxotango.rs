use taxotangolib::*;

use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use burn::prelude::*;

fn main() {
    env_logger::init();

    let nodes_file = "/mnt/data/data/nt/taxdmp/nodes.dmp";
    let names_file = "/mnt/data/data/nt/taxdmp/names.dmp";

    // let mut batch_gen = build_taxonomy_graph(nodes_file, names_file);

    // log::debug!("Generating first test batch");

    // let batch = batch_gen.generate_batch();
    // println!("{:?}", batch);

    let config = PoincareTaxonomyEmbeddingModelConfig {
        taxonomy_size: 10,
        embedding_size: 8,
    };

    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    let device = burn::backend::wgpu::WgpuDevice::default();

    let nn: PoincareTaxonomyEmbeddingModel<Wgpu> = config.init(&device);

    let output = nn.forward(Tensor::<MyBackend, 2, Int>::from_data(
        [[1, 2], [3, 4], [1, 6]],
        &device,
    ));

    println!("{}", output);

    // let device = Default::default();
    // Creation of two tensors, the first with explicit values and the second one with ones, with the same shape as the first
    // let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]], &device);
    // let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);

    // Print the element-wise addition (done with the WGPU backend) of the two tensors.
    // println!("{}", tensor_1 + tensor_2);
}
