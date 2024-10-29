use burn::{
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::adaptor::OptimizerAdaptor,
    optim::decay::{WeightDecay, WeightDecayConfig},
    optim::momentum::{Momentum, MomentumConfig, MomentumState},
    optim::SimpleOptimizer,
    prelude::*,
    record::Record,
    tensor::backend::AutodiffBackend,
    LearningRate,
};

use crate::l2_norm;

#[derive(Config)]
pub struct RiemannianSgdConfig {
    /// [Weight decay](WeightDecayConfig) config.
    weight_decay: Option<WeightDecayConfig>,
    /// [Momentum](MomentumConfig) config.
    momentum: Option<MomentumConfig>,
    /// [Gradient Clipping](GradientClippingConfig) config.
    gradient_clipping: Option<GradientClippingConfig>,
}

#[derive(Clone)]
pub struct RiemannianSgd<B: Backend> {
    momentum: Option<Momentum<B>>,
    weight_decay: Option<WeightDecay<B>>,
}

/// State of [RiemannianSgd](RiemannianSgd).
#[derive(Record, Clone)]
pub struct RiemannianSgdState<B: Backend, const D: usize> {
    momentum: Option<MomentumState<B, D>>,
}

impl<B: Backend, const D: usize> RiemannianSgdState<B, D> {
    pub fn new(momentum: Option<MomentumState<B, D>>) -> Self {
        Self { momentum }
    }
}

impl RiemannianSgdConfig {
    /// Creates a new [RiemannianSgdConfig](RiemannianSgdConfig) with default values.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<RiemannianSgd<B::InnerBackend>, M, B> {
        let momentum = self.momentum.as_ref().map(Momentum::new);
        let weight_decay = self.weight_decay.as_ref().map(WeightDecay::new);

        println!("Weight decay? {}", weight_decay.is_some());
        println!("Momentum? {}", momentum.is_some());
        println!("Grad Clipping? {}", self.gradient_clipping.is_some());

        let mut optim = OptimizerAdaptor::from(RiemannianSgd {
            momentum,
            weight_decay,
        });

        if let Some(config) = &self.gradient_clipping {
            optim = optim.with_grad_clipping(config.init());
        }

        optim
    }
}

impl<B: Backend> RiemannianSgd<B> {
    fn mobius_add<const D: usize>(&self, x: Tensor<B, D>, y: Tensor<B, D>) -> Tensor<B, D> {
        let x2 = x.clone().powf_scalar(2.0).sum_dim(D - 1);
        let y2 = y.clone().powf_scalar(2.0).sum_dim(D - 1);
        let xy = (x.clone() * y.clone()).sum_dim(D - 1);

        let ones = Tensor::<B, D>::ones_like(&x2);

        let num = ((xy.clone().mul_scalar(2.0).add_scalar(1.0) + y2.clone()) * x)
            + ((ones - x2.clone()) * y);
        let denom = xy.mul_scalar(2.0).add_scalar(1.0) + (x2 * y2);

        num / denom.clamp_min(1e-15)
    }

    fn expm<const D: usize>(&self, p: Tensor<B, D>, u: Tensor<B, D>) -> Tensor<B, D> {
        // Calculate the norm of u
        let norm = u
            .clone()
            .powf_scalar(2.0)
            .sum_dim(D - 1)
            .sqrt()
            .clamp_min(1e-10)
            .unsqueeze();

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

    // Custom gradient scaling for the Riemannian manifold
    fn grad<const D: usize>(&self, p: Tensor<B, D>, grad: Tensor<B, D>) -> Tensor<B, D> {
        // let p_sqnorm = p.powf_scalar(2.0).sum_dim(D - 1);
        // let ones = Tensor::<B, D>::ones_like(&p_sqnorm);
        // grad * ((ones - p_sqnorm).powf_scalar(2.0).div_scalar(4.0))

        let p_sqnorm = p.powf_scalar(2.0).sum_dim(D - 1);
        let scaling = (Tensor::<B, D>::ones_like(&p_sqnorm).sub(p_sqnorm))
            .powf_scalar(2.0)
            .div_scalar(4.0);
            // .clamp_min(1e-12);
        grad * scaling
    }

    pub fn project_to_manifold<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // Compute norms of embeddings
        let norms = l2_norm(x.clone()).unsqueeze();

        // Create a mask for embeddings with norms >= 1
        let gte_mask = norms.clone().greater_equal_elem(1.0).expand(x.shape());

        // Scale embeddings that are outside the unit ball
        let scaled_x = x.clone() / norms;
        let adjusted_x = scaled_x * (1.0 - 1e-5); // Subtract a small value to ensure it's inside

        // Replace embeddings outside the unit ball with adjusted ones
        x.mask_where(gte_mask, adjusted_x)
    }
}
impl<B: Backend> SimpleOptimizer<B> for RiemannianSgd<B> {
    type State<const D: usize> = RiemannianSgdState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        mut grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let mut state_momentum = None;

        if let Some(state) = state {
            state_momentum = state.momentum;
        }

        if let Some(weight_decay) = &self.weight_decay {
            grad = weight_decay.transform(grad, tensor.clone());
        }

        // Apply the custom Riemannian gradient scaling
        grad = self.grad(tensor.clone(), grad);

        if let Some(momentum) = &self.momentum {
            let (grad_out, state) = momentum.transform(grad, state_momentum);
            state_momentum = Some(state);
            grad = grad_out;
        }

        let state = RiemannianSgdState::new(state_momentum);

        let delta = grad.mul_scalar(-lr);

        // Update parameters using the exponential map
        let updated_tensor = self.expm(tensor, delta);

        let projected_tensor = self.project_to_manifold(updated_tensor);

        (projected_tensor, Some(state))
        // (updated_tensor, Some(state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &B::Device) -> Self::State<D> {
        state.momentum = state.momentum.map(|state| state.to_device(device));
        state
    }
}
