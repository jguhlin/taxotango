use core::num;

use burn::prelude::*;

use super::l2norm::L2Norm;

#[derive(Module, Debug, Clone)]
pub struct PoincareDistance {
    pub l2_norm: L2Norm,
    pub eps: f32,
    pub clamp_min: f32,
    pub clamp_max: f32,
}

impl PoincareDistance {
    pub fn new() -> Self {
        Self {
            l2_norm: L2Norm::new(),
            eps: 1e-12,
            clamp_min: 1e-12,
            clamp_max: f32::MAX,
        }
    }

    pub fn forward<B: Backend>(&self, u: Tensor<B, 3>, v: Tensor<B, 3>) -> Tensor<B, 2> {
        let u_norm = self.l2_norm.forward(u.clone());
        let v_norm = self.l2_norm.forward(v.clone());

        let euclidean_distance = self.l2_norm.forward(u - v).powf_scalar(2.0);

        let numerator = euclidean_distance;
        let numerator = numerator.add_scalar(self.eps);
        let ones = Tensor::<B, 3>::ones_like(&u_norm);
        let denominator = (ones.clone() - u_norm.clone().powf_scalar(2.0))
            * (ones - v_norm.clone().powf_scalar(2.0));
        let denominator = denominator.clamp_min(self.eps);

        let distance = numerator / denominator;
        let distance = distance.mul_scalar(2.0).add_scalar(1.0);

        let distance = distance.clamp(self.clamp_min, self.clamp_max);

        // println!("{}", distance);

        let distance = distance.squeeze(2);
        
        // println!("{}", distance);

        self.acosh(distance)
    }

    pub fn acosh<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let x = x.clamp_min(1.0);
        let x_squared = x.clone().powf_scalar(2.0).sub_scalar(1.0);
        // This destroys equals being 0, but that's really fine...
        let inside_sqrt = x_squared.clamp_min(self.clamp_min);
        let sqrt_term = inside_sqrt.sqrt();
        (x + sqrt_term).log()
    }
}

#[cfg(test)]
mod tests {
    use burn::prelude::*;
    use burn::backend::{Autodiff, Wgpu};

    use super::*;

    #[test]
    fn test_poincare_distance() {
        let device = burn::backend::wgpu::WgpuDevice::default();

        let poincare_distance = PoincareDistance::new();

        let u = TensorData::new(vec![1.0, 5.0, 4.0], vec![1, 1, 3]);
        let u = Tensor::<Wgpu, 3>::from_data(u, &device);

        let v = TensorData::new(vec![1.0, 5.0, 4.0], vec![1, 1, 3]);
        let v = Tensor::<Wgpu, 3>::from_data(v, &device);

        let distance = poincare_distance.forward(u, v);
        let distance = distance.into_scalar();

        assert_eq!(distance, 0.0);

        // Next test

        let u = TensorData::new(vec![1.0, 5.0, 2.0], vec![1, 1, 3]);
        let u = Tensor::<Wgpu, 3>::from_data(u, &device);

        let v = TensorData::new(vec![4.0, 2.0, 4.0], vec![1, 1, 3]);
        let v = Tensor::<Wgpu, 3>::from_data(v, &device);

        let distance = poincare_distance.forward(u, v);

        let distance = distance.into_scalar();

        assert_eq!(distance, 0.29339474);
    }

    #[test]
    fn test_acosh() {
        let device = burn::backend::wgpu::WgpuDevice::default();

        let poincare_distance = PoincareDistance::new();

        // Values to test
        let values = vec![1.0, 5.0, 10.0, 1.2];
        let results = vec![0.0, 2.2924316, 2.9932227, 0.6223626];

        // Results

        for (i, value) in values.iter().enumerate() {
            let x = TensorData::new(vec![*value], vec![1]);
            let x = Tensor::<Wgpu, 1>::from_data(x, &device);

            let y = poincare_distance.acosh(x);

            let y = y.into_scalar();

            assert_eq!(y, results[i]);
        }
    }
}

