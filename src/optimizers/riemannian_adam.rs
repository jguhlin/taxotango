
/// Example usage of the Riemannian Adam optimizer.
///
/// ```rust
/// use burn::{
///     module::Module,
///     tensor::backend::Backend,
///     optim::Optimizer,
///     train::TrainOutput,
///     module::ModuleTransformer,
/// };
///
/// // Define your model that uses Poincaré embeddings
/// struct MyModel<B: Backend> {
///     embeddings: Tensor<B, 2>, // Example embedding tensor
/// }
///
/// impl<B: Backend> Module<B> for MyModel<B> {
///     type Input = Tensor<B, 2>;
///     type Output = Tensor<B, 2>;
///
///     fn forward(&self, input: Self::Input) -> Self::Output {
///         // Model forward logic here
///         input.matmul(self.embeddings.clone())
///     }
/// }
///
/// // Training loop
/// fn train<B: AutodiffBackend>(model: &mut MyModel<B>, data: Vec<Tensor<B, 2>>) {
///     // Initialize optimizer configuration
///     let optimizer_config = RiemannianAdamConfig {
///         weight_decay: None,
///         gradient_clipping: None,
///         beta1: 0.9,
///         beta2: 0.999,
///         eps: 1e-8,
///     };
///
///     // Initialize optimizer
///     let mut optimizer = optimizer_config.init::<B, MyModel<B>>();
///
///     // Training loop
///     for epoch in 0..10 {
///         for batch in &data {
///             // Zero gradients
///             optimizer.zero_grad(model);
///
///             // Forward pass
///             let output = model.forward(batch.clone());
///
///             // Compute loss (example loss function)
///             let loss = output.mean();
///
///             // Backward pass
///             optimizer.backward(&loss);
///
///             // Update model parameters
///             optimizer.step(model);
///         }
///     }
/// }
/// ```

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
use std::marker::PhantomData;

/// Computes the L2 norm of a tensor along the last dimension.
fn l2_norm<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.powf_scalar(2.0).sum_dim(D - 1).sqrt()
}

/// Configuration for the Riemannian Adam optimizer.
#[derive(Config)]
pub struct RiemannianAdamConfig {
    /// [Weight decay](WeightDecayConfig) config.
    pub weight_decay: Option<WeightDecayConfig>,
    /// [Gradient Clipping](GradientClippingConfig) config.
    pub gradient_clipping: Option<GradientClippingConfig>,
    /// Exponential decay rate for the first moment estimates (default: 0.9).
    pub beta1: f64,
    /// Exponential decay rate for the second moment estimates (default: 0.999).
    pub beta2: f64,
    /// Small constant for numerical stability (default: 1e-8).
    pub eps: f64,
}

impl RiemannianAdamConfig {
    /// Initializes the Riemannian Adam optimizer with the given backend and module.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<RiemannianAdam<B::InnerBackend>, M, B> {
        let weight_decay = self.weight_decay.as_ref().map(WeightDecay::new);

        let mut optim = OptimizerAdaptor::from(RiemannianAdam {
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

/// Riemannian Adam optimizer for Poincaré embeddings.
#[derive(Clone)]
pub struct RiemannianAdam<B: Backend> {
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: Option<WeightDecay<B>>,
    _marker: PhantomData<B>,
}

/// State of the Riemannian Adam optimizer.
#[derive(Record, Clone)]
pub struct RiemannianAdamState<B: Backend, const D: usize> {
    m: Tensor<B, D>,
    v: Tensor<B, D>,
    t: usize,
}

impl<B: Backend, const D: usize> RiemannianAdamState<B, D> {
    /// Creates a new `RiemannianAdamState`.
    pub fn new(m: Tensor<B, D>, v: Tensor<B, D>, t: usize) -> Self {
        Self { m, v, t }
    }
}

impl<B: Backend> RiemannianAdam<B> {
    /// Performs Möbius addition of two tensors in the Poincaré ball model.
    fn mobius_add<const D: usize>(&self, x: Tensor<B, D>, y: Tensor<B, D>) -> Tensor<B, D> {
        let x2 = x.clone().powf_scalar(2.0).sum_dim(D - 1).unsqueeze();
        let y2 = y.clone().powf_scalar(2.0).sum_dim(D - 1).unsqueeze();
        let xy = (x.clone() * y.clone()).sum_dim(D - 1).unsqueeze();

        let ones = Tensor::<B, D>::ones_like(&x2);

        let num = ((xy.clone().mul_scalar(2.0).add_scalar(1.0) + y2.clone()) * x)
            + ((ones - x2.clone()) * y);
        let denom = xy.mul_scalar(2.0).add_scalar(1.0) + (x2 * y2);

        num / denom.clamp_min(1e-15)
    }

    /// Exponential map from the tangent space at point `p` to the manifold.
    fn expm<const D: usize>(&self, p: Tensor<B, D>, u: Tensor<B, D>) -> Tensor<B, D> {

        let device = p.device();

        let norm = u
            .clone()
            .powf_scalar(2.0)
            .sum_dim(D - 1)
            .sqrt()
            .clamp_min(1e-10)
            .unsqueeze();

        let p_sqnorm = p.clone().powf_scalar(2.0).sum_dim(D - 1).unsqueeze();
        let ones = Tensor::<B, D>::ones_like(&p_sqnorm);
        let lambda_p = Tensor::<B, D>::from_floats([2.0], &device) / (ones - p_sqnorm).clamp_min(1e-15);

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
        let scaling = ((Tensor::<B, D>::ones_like(&p_sqnorm) - p_sqnorm)
            .powf_scalar(2.0)
            * 0.25)
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

impl<B: Backend> SimpleOptimizer<B> for RiemannianAdam<B> {
    type State<const D: usize> = RiemannianAdamState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
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

        if let Some(weight_decay) = &self.weight_decay {
            grad = weight_decay.transform(grad, tensor.clone());
        }

        // Apply Riemannian gradient scaling
        grad = self.grad(tensor.clone(), grad);

        // Update biased first moment estimate
        m = m.mul_scalar(self.beta1).add(grad.clone().mul_scalar(1.0 - self.beta1));

        // Update biased second raw moment estimate
        let grad_squared = grad.clone().powf_scalar(2.0);
        v = v.mul_scalar(self.beta2).add(grad_squared.mul_scalar(1.0 - self.beta2));

        // Compute bias-corrected first moment estimate
        let beta1_t = self.beta1.powi(t as i32);
        let m_hat = m.clone().div_scalar(1.0 - beta1_t);

        // Compute bias-corrected second moment estimate
        let beta2_t = self.beta2.powi(t as i32);
        let v_hat = v.clone().div_scalar(1.0 - beta2_t);

        // Compute the update direction
        let v_hat_sqrt = v_hat.sqrt().add_scalar(self.eps);
        let delta = m_hat.div(v_hat_sqrt).mul_scalar(-lr);

        // Update parameters using the exponential map
        let updated_tensor = self.expm(tensor, delta);

        // Project back onto the manifold
        let projected_tensor = self.project_to_manifold(updated_tensor);

        let new_state = RiemannianAdamState::new(m, v, t);

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
