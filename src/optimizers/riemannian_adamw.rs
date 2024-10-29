use crate::l2_norm;
use burn::{
    config::Config,
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::adaptor::OptimizerAdaptor,
    optim::decay::{WeightDecay, WeightDecayConfig},
    optim::SimpleOptimizer,
    record::Record,
    tensor::backend::AutodiffBackend,
    tensor::{backend::Backend, Tensor},
    LearningRate,
};

use std::marker::PhantomData;

const EPS: f64 = 1e-8;

/// Configuration for the Riemannian AdamW optimizer.
#[derive(Config)]
pub struct RiemannianAdamWConfig {
    /// Weight decay coefficient (default: 0.01).
    #[config(default = 0.01)]
    pub weight_decay: f64,
    /// [Gradient Clipping](GradientClippingConfig) config.
    pub gradient_clipping: Option<GradientClippingConfig>,
    /// Exponential decay rate for the first moment estimates (default: 0.9).
    #[config(default = 0.9)]
    pub beta1: f64,
    /// Exponential decay rate for the second moment estimates (default: 0.999).
    #[config(default = 0.999)]
    pub beta2: f64,
    /// Small constant for numerical stability (default: 1e-8).
    #[config(default = 1e-8)]
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

        // if let Some(config) = &self.gradient_clipping {
            // optim = optim.with_grad_clipping(config.init());
        // }

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
        let x2 = x.clone().powf_scalar(2.0).sum_dim(D - 1);
        let y2 = y.clone().powf_scalar(2.0).sum_dim(D - 1);
        let xy = (x.clone() * y.clone()).sum_dim(D - 1);
    
        let ones = Tensor::<B, D>::ones_like(&x2);
    
        let num = ((xy.clone() * 2.0 + 1.0 + y2.clone()) * x.clone())
            + ((ones.clone() - x2.clone()) * y.clone());
        let denom = xy * 2.0 + 1.0 + x2 * y2;
    
        num / denom.clamp_min(EPS)
    }
    

    /// Exponential map from the tangent space at point `p` to the manifold.
    fn expm<const D: usize>(&self, p: Tensor<B, D>, u: Tensor<B, D>) -> Tensor<B, D> {
        let device = p.device();
    
        let norm = u
            .clone()
            .powf_scalar(2.0)
            .sum_dim(D - 1)
            .sqrt()
            .clamp_min(EPS);
        // `norm` shape: `[N1, N2, ..., 1]`
    
        let p_sqnorm = p.clone().powf_scalar(2.0).sum_dim(D - 1);
        // `p_sqnorm` shape: `[N1, N2, ..., 1]`
    
        let ones = Tensor::<B, D>::ones_like(&p_sqnorm);
        let twos = Tensor::<B, D>::from_floats([[2.0]], &device).expand(p_sqnorm.shape());
        let lambda_p = twos / (ones - p_sqnorm).clamp_min(EPS);
    
        let scaled_u = ((lambda_p.clone() * norm.clone() * 0.5).tanh() * u.clone())
            / norm.clone().clamp_min(EPS);
    
        self.mobius_add(p, scaled_u)
    }
    
    /// Scales the Euclidean gradient to obtain the Riemannian gradient.
    fn grad<const D: usize>(&self, p: Tensor<B, D>, grad: Tensor<B, D>) -> Tensor<B, D> {
        let p_sqnorm = p.powf_scalar(2.0).sum_dim(D - 1); // Specify dimension here
        let ones = Tensor::<B, D>::ones_like(&p_sqnorm);
        let scaling = ((ones - p_sqnorm).powf_scalar(2.0) * 0.25).clamp_min(EPS);

        grad * scaling
    }

    pub fn project_to_manifold<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // clamp_min removed
        let norms = l2_norm(x.clone());
    
        let scaled_x = x.clone() / norms.clone().add_scalar(EPS);
        let mut projected_x = x.clone().mask_where(norms.clone().greater_equal_elem(1.0).expand(x.clone().shape()), scaled_x);

        let mut norms = l2_norm(projected_x.clone());
        while norms.clone().greater_elem(1.0).any().into_scalar() {
            projected_x = self.project_to_manifold(projected_x);
            norms = l2_norm(projected_x.clone());
        }
 
        projected_x
    }
    
}

impl<B: Backend> SimpleOptimizer<B> for RiemannianAdamW<B> {
    type State<const D: usize> = RiemannianAdamWState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
    
        let (mut m, mut v, mut t) = if let Some(state) = state {
            (state.m, state.v, state.t)
        } else {
            let zeros = Tensor::<B, D>::zeros_like(&tensor);
            (zeros.clone(), zeros.clone(), 0)
        };
    
        t += 1;
    
        // Apply weight decay to the gradient
        let mut grad = grad + tensor.clone() * self.weight_decay;
    
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
        let delta = m_hat / v_hat_sqrt * (-lr);
    
        // Update parameters using the exponential map
        let updated_tensor = self.expm(tensor.clone(), delta);
    
        // Project back onto the manifold
        let projected_tensor = self.project_to_manifold(updated_tensor);
    
        let new_state = RiemannianAdamWState::new(m, v, t);
    
        (projected_tensor, Some(new_state))
    }

    fn to_device<const D: usize>(
        mut state: Self::State<D>,
        device: &<B as Backend>::Device,
    ) -> Self::State<D> {
        state.m = state.m.to_device(device);
        state.v = state.v.to_device(device);
        state
    }
}