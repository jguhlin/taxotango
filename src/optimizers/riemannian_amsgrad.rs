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

use crate::l2_norm;

use std::marker::PhantomData;

/// Configuration for the Riemannian AMSGrad optimizer.
#[derive(Config)]
pub struct RiemannianAMSGradConfig {
    /// [Weight decay](WeightDecayConfig) config.
    pub weight_decay: Option<WeightDecayConfig>,
    /// [Gradient Clipping](GradientClippingConfig) config.
    pub gradient_clipping: Option<GradientClippingConfig>,
    /// Exponential decay rate for the first moment estimates (default: 0.9).
    #[config(default = 0.9)]
    pub beta1: f64,
    /// Exponential decay rate for the second moment estimates (default: 0.999).
    #[config(default = 0.99)]
    pub beta2: f64,
    /// Small constant for numerical stability (default: 1e-8).
    #[config(default = 1e-8)]
    pub eps: f64,
}

impl RiemannianAMSGradConfig {
    /// Initializes the Riemannian AMSGrad optimizer.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<RiemannianAMSGrad<B::InnerBackend>, M, B> {
        let weight_decay = self.weight_decay.as_ref().map(WeightDecay::new);

        let mut optim = OptimizerAdaptor::from(RiemannianAMSGrad {
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay,
            _marker: PhantomData,
        });

        if let Some(config) = &self.gradient_clipping {
            optim = optim.with_grad_clipping(config.init());
        }

        optim
    }
}

/// Riemannian AMSGrad optimizer for Poincaré embeddings.
#[derive(Clone)]
pub struct RiemannianAMSGrad<B: Backend> {
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: Option<WeightDecay<B>>,
    _marker: PhantomData<B>,
}

/// State of the Riemannian AMSGrad optimizer.
#[derive(Record, Clone)]
pub struct RiemannianAMSGradState<B: Backend, const D: usize> {
    m: Tensor<B, D>,
    v: Tensor<B, D>,
    v_hat: Tensor<B, D>,
    t: usize,
}

impl<B: Backend, const D: usize> RiemannianAMSGradState<B, D> {
    /// Creates a new `RiemannianAMSGradState`.
    pub fn new(m: Tensor<B, D>, v: Tensor<B, D>, v_hat: Tensor<B, D>, t: usize) -> Self {
        Self { m, v, v_hat, t }
    }
}

impl<B: Backend> RiemannianAMSGrad<B> {
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

    fn expm<const D: usize>(&self, p: Tensor<B, D>, u: Tensor<B, D>) -> Tensor<B, D> {
        // Calculate the norm of u
        let norm = u
            .clone()
            .powf_scalar(2.0)
            .sum_dim(D - 1)
            .sqrt()
            .clamp_min(1e-10);

        // Calculate lambda_x(p), which is a scaling factor based on the point p
        let p_sqnorm = p.clone().powf_scalar(2.0).sum_dim(D - 1);
        let ones = Tensor::<B, D>::ones_like(&p_sqnorm);
        let twos = Tensor::<B, D>::full(p_sqnorm.shape(), 2.0, &p.device());
        let lambda_x = twos / (ones.sub(p_sqnorm)).clamp(1e-15, f64::INFINITY);

        // Scale u by tanh(0.5 * lambda_x(p) * norm) / norm
        let scaled_u = (lambda_x.mul_scalar(0.5) * norm.clone()).tanh() * u / norm.clamp_min(1e-15);

        // Perform the Möbius addition
        self.mobius_add(p, scaled_u)
    }

    /// Scales the Euclidean gradient to obtain the Riemannian gradient.
    fn grad<const D: usize>(&self, p: Tensor<B, D>, grad: Tensor<B, D>) -> Tensor<B, D> {
        let p_sqnorm = p.powf_scalar(2.0).sum_dim(D - 1).unsqueeze();
        let scaling = ((Tensor::<B, D>::ones_like(&p_sqnorm) - p_sqnorm).powf_scalar(2.0) * 0.25)
            .clamp_min(1e-12);

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

impl<B: Backend> SimpleOptimizer<B> for RiemannianAMSGrad<B> {
    type State<const D: usize> = RiemannianAMSGradState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        mut grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let (mut m, mut v, mut v_hat, mut t) = if let Some(state) = state {
            (state.m, state.v, state.v_hat, state.t)
        } else {
            let zeros = Tensor::<B, D>::zeros_like(&tensor);
            (zeros.clone(), zeros.clone(), zeros.clone(), 0)
        };

        t += 1;

        if let Some(weight_decay) = &self.weight_decay {
            grad = weight_decay.transform(grad, tensor.clone());
        }

        // Apply Riemannian gradient scaling
        grad = self.grad(tensor.clone(), grad);

        // Update biased first moment estimate
        m = m * self.beta1 + grad.clone() * (1.0 - self.beta1);

        // Update biased second raw moment estimate
        let grad_squared = grad.clone().powf_scalar(2.0);
        v = v * self.beta2 + grad_squared * (1.0 - self.beta2);

        // Compute v_hat (element-wise maximum of v_hat and v)
        v_hat = v_hat.max_pair(v.clone());

        // Compute bias-corrected first moment estimate
        let beta1_t = self.beta1.powi(t as i32);
        let m_hat = m.clone() / (1.0 - beta1_t);

        // Compute bias-corrected second moment estimate
        let beta2_t = self.beta2.powi(t as i32);
        let v_hat_corr = v_hat.clone() / (1.0 - beta2_t);

        // Compute the update direction
        let v_hat_sqrt = v_hat_corr.sqrt() + self.eps;
        let delta = m_hat / v_hat_sqrt * -lr;

        // Update parameters using the exponential map
        let updated_tensor = self.expm(tensor, delta);

        // Project back onto the manifold
        let projected_tensor = self.project_to_manifold(updated_tensor);

        let new_state = RiemannianAMSGradState::new(m, v, v_hat, t);

        (projected_tensor, Some(new_state))
    }

    fn to_device<const D: usize>(
        mut state: Self::State<D>,
        device: &<B as burn::tensor::backend::Backend>::Device,
    ) -> Self::State<D> {
        state.m = state.m.to_device(device);
        state.v = state.v.to_device(device);
        state.v_hat = state.v_hat.to_device(device);
        state
    }
}
