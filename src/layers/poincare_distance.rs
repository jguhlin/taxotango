use core::num;

use burn::prelude::*;

use super::l2norm::*;

const EPS: f32 = 1e-8;
const CLAMP_MIN: f32 = 1e-8;
const CLAMP_MAX: f32 = f32::MAX;

pub fn poincare_distance<B: Backend>(u: Tensor<B, 3>, v: Tensor<B, 3>) -> Tensor<B, 2> {
    let u_norm = l2_norm(u.clone());
    let v_norm = l2_norm(v.clone());

    let u_norm_sq = u_norm.clone().powf_scalar(2.0).clamp_max(1.0 - EPS);
    let v_norm_sq = v_norm.clone().powf_scalar(2.0).clamp_max(1.0 - EPS);

    // println!("U Norm_sq: {}", u_norm_sq);
    // println!("V Norm_sq: {}", v_norm_sq);

    let euclidean_distance_sq = l2_norm(u - v).powf_scalar(2.0);

    // println!("Euclidean Distance: {}", euclidean_distance_sq);

    let numerator = euclidean_distance_sq.add_scalar(EPS);

    let ones = Tensor::<B, 3>::ones_like(&u_norm);
    let denominator = (ones.clone() - u_norm_sq) * (ones - v_norm_sq);
    let denominator = denominator.add_scalar(EPS);

    // println!("Numerator: {}", numerator);
    // println!("Denominator: {}", denominator);

    let mut distance = numerator / denominator;
    // println!("Distance: {}", distance);

    distance = distance.mul_scalar(2.0).add_scalar(1.0);

    let distance = distance.clamp_min(1.0 + EPS);

    let distance = distance.squeeze(2);

    // println!("Distances before acosh: {}", distance);
    
    acosh(distance)
}

pub fn acosh<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    // Clamp x to be at least 1 + EPS to ensure x^2 - 1 > 0
    let x = x.clamp_min(1.0 + EPS);

    // Compute x_squared = x^2 - 1
    let x_squared = x.clone().powf_scalar(2.0).sub_scalar(1.0);

    let x_squared = x_squared.clamp_min(CLAMP_MIN);

    // debug thing
    if x_squared.clone().lower_equal_elem(0.0).any().into_scalar() {
        panic!("x_squared <= 0");
    }

    // Since x >= 1 + EPS, x_squared > 0, so sqrt is valid
    let sqrt_term = x_squared.sqrt();

    // Compute acosh(x) = ln(x + sqrt(x^2 - 1))
    (x + sqrt_term).log()
}

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

