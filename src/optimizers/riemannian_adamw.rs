use burn::{
    tensor::{backend::Backend, Tensor},
    optim::decay::{WeightDecay, WeightDecayConfig},
    optim::adaptor::OptimizerAdaptor,
    optim::SimpleOptimizer,
    grad_clipping::GradientClippingConfig,
    tensor::backend::AutodiffBackend,
    module::AutodiffModule,
    record::Record,
    config::Config,
    LearningRate,
};
use crate::l2_norm;

use std::marker::PhantomData;


/// Configuration for the Riemannian AdamW optimizer.
#[derive(Config)]
pub struct RiemannianAdamWConfig {
    /// Weight decay coefficient (default: 0.01).
    pub weight_decay: f64,
    /// [Gradient Clipping](GradientClippingConfig) config.
    pub gradient_clipping: Option<GradientClippingConfig>,
    /// Exponential decay rate for the first moment estimates (default: 0.9).
    pub beta1: f64,
    /// Exponential decay rate for the second moment estimates (default: 0.999).
    pub beta2: f64,
    /// Small constant for numerical stability (default: 1e-8).
    pub eps: f64,
}

impl RiemannianAdamWConfig {
    /// Initializes the Riemannian AdamW optimizer.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<RiemannianAdamW<B::InnerBackend>, M, B> {
        let mut optim = OptimizerAdaptor::from(RiemannianAdamW {
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
            _marker: PhantomData,
        });

        if let Some(config) = &self.gradient_clipping {
            optim = optim.with_grad_clipping(config.init());
        }

        optim
    }
}

/// Riemannian AdamW optimizer for Poincaré embeddings.
#[derive(Clone)]
pub struct RiemannianAdamW<B: Backend> {
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    _marker: PhantomData<B>,
}

/// State of the Riemannian AdamW optimizer.
#[derive(Record, Clone)]
pub struct RiemannianAdamWState<B: Backend, const D: usize> {
    m: Tensor<B, D>,
    v: Tensor<B, D>,
    t: usize,
}

impl<B: Backend, const D: usize> RiemannianAdamWState<B, D> {
    /// Creates a new `RiemannianAdamWState`.
    pub fn new(m: Tensor<B, D>, v: Tensor<B, D>, t: usize) -> Self {
        Self { m, v, t }
    }
}

impl<B: Backend> RiemannianAdamW<B> {
    /// Performs Möbius addition of two tensors in the Poincaré ball model.
    fn mobius_add<const D: usize>(&self, x: Tensor<B, D>, y: Tensor<B, D>) -> Tensor<B, D> {
        let x2 = x.clone().powf_scalar(2.0).sum_dim(D - 1).unsqueeze();
        let y2 = y.clone().powf_scalar(2.0).sum_dim(D - 1).unsqueeze();
        let xy = (x.clone() * y.clone()).sum_dim(D - 1).unsqueeze();

        let ones = Tensor::<B, D>::ones_like(&x2);

        let num = ((xy.clone() * 2.0 + 1.0 + y2.clone()) * x) + ((ones - x2.clone()) * y);
        let denom = xy * 2.0 + 1.0 + x2 * y2;

        num / denom.clamp_min(1e-15)
    }

    /// Exponential map from the tangent space at point `p` to the manifold.
    fn expm<const D: usize>(&self, p: Tensor<B, D>, u: Tensor<B, D>) -> Tensor<B, D> {

        let device: <B as Backend>::Device = p.device();

        let norm = u
            .clone()
            .powf_scalar(2.0)
            .sum_dim(D - 1)
            .sqrt()
            .clamp_min(1e-10)
            .unsqueeze();

        let p_sqnorm = p.clone().powf_scalar(2.0).sum_dim(D - 1).unsqueeze();
        let ones = Tensor::<B, D>::ones_like(&p_sqnorm);
        let twos = Tensor::<B, D>::from_floats([2.0], &device).expand(p_sqnorm.shape());
        let lambda_p = twos / (ones - p_sqnorm).clamp_min(1e-15);

        let scaled_u = ((lambda_p.clone() * norm.clone() * 0.5).tanh() * u)
            / norm.clamp_min(1e-15);

        self.mobius_add(p, scaled_u)
    }

    /// Scales the Euclidean gradient to obtain the Riemannian gradient.
    fn grad<const D: usize>(&self, p: Tensor<B, D>, grad: Tensor<B, D>) -> Tensor<B, D> {
        let p_sqnorm = p
            .powf_scalar(2.0)
            .sum_dim(D - 1)
            .unsqueeze();
        let ones = Tensor::<B, D>::ones_like(&p_sqnorm);
        let scaling = ((ones - p_sqnorm).powf_scalar(2.0) * 0.25).clamp_min(1e-12);

        grad * scaling
    }

    /// Projects points back onto the manifold if they have moved outside.
    pub fn project_to_manifold<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let norms = l2_norm(x.clone()).unsqueeze();

        let gte_mask = norms.clone().greater_equal_elem(1.0).expand(x.shape());

        let scaled_x = x.clone() / norms;
        let adjusted_x = scaled_x * (1.0 - 1e-5);

        x.mask_where(gte_mask, adjusted_x)
    }
}

impl<B: Backend> SimpleOptimizer<B> for RiemannianAdamW<B> {
    type State<const D: usize> = RiemannianAdamWState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        mut tensor: Tensor<B, D>,
        mut grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let (mut m, mut v, mut t) = if let Some(state) = state {
            (state.m, state.v, state.t)
        } else {
            let zeros = Tensor::<B, D>::zeros_like(&tensor);
            (zeros.clone(), zeros.clone(), 0)
        };

        t += 1;

        // Apply weight decay directly to the parameters (decoupled)
        tensor = tensor * (1.0 - lr * self.weight_decay);

        // Apply Riemannian gradient scaling
        grad = self.grad(tensor.clone(), grad);

        // Update biased first moment estimate
        m = m * self.beta1 + grad.clone() * (1.0 - self.beta1);

        // Update biased second raw moment estimate
        let grad_squared = grad.clone().powf_scalar(2.0);
        v = v * self.beta2 + grad_squared * (1.0 - self.beta2);

        // Compute bias-corrected first moment estimate
        let beta1_t = self.beta1.powi(t as i32);
        let m_hat = m.clone() / (1.0 - beta1_t);

        // Compute bias-corrected second moment estimate
        let beta2_t = self.beta2.powi(t as i32);
        let v_hat = v.clone() / (1.0 - beta2_t);

        // Compute the update direction
        let v_hat_sqrt = v_hat.sqrt() + self.eps;
        let delta = m_hat / v_hat_sqrt * -lr;

        // Update parameters using the exponential map
        let updated_tensor = self.expm(tensor, delta);

        // Project back onto the manifold
        let projected_tensor = self.project_to_manifold(updated_tensor);

        let new_state = RiemannianAdamWState::new(m, v, t);

        (projected_tensor, Some(new_state))
    }

    fn to_device<const D: usize>(
        mut state: Self::State<D>,
        device: &<B as burn::tensor::backend::Backend>::Device,
    ) -> Self::State<D> {
        state.m = state.m.to_device(device);
        state.v = state.v.to_device(device);
        state
    }
}
