use burn::prelude::*;

#[derive(Module, Debug, Clone)]
pub struct L2Norm {
    pub eps: f32,

}

impl L2Norm {
    pub fn new() -> Self {
        Self {
            eps: 1e-12,
        }
    }

    pub fn forward<B: Backend, const N: usize>(&self, x: Tensor<B, N>) -> Tensor<B, N> {
        // Because you can't take the sqrt of 0
        x.powf_scalar(2.0)
            .sum_dim(N - 1)
            .clamp(self.eps, f32::MAX)
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use burn::prelude::*;
    use burn::backend::{Autodiff, Wgpu};

    use super::*;

    #[test]
    fn test_l2norm() {
        let device = burn::backend::wgpu::WgpuDevice::default();

        let l2norm = L2Norm::new();

        let x = TensorData::new(vec![1.0, 5.0, 4.0], vec![1, 1, 3]);
        let x = Tensor::<Wgpu, 3>::from_data(x, &device);

        let y = l2norm.forward(x.clone());
        
        let y = y.into_scalar();

        assert_eq!(y, 6.48074069841);

    }
}