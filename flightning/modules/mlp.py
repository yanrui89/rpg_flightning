from typing import NamedTuple
from typing import Union

import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron.

    Example creating a 2-layer MLP with 2 input features, 3 hidden units,
    and 1 output unit, and weights initialized with small variance scaling.
    Note that the bias is always initialized with zeros.

    >>> network = MLP([2, 3, 1])
    >>> key = jax.random.key(0)
    >>> key0, key1, key2 = jax.random.split(key, 3)
    >>> x_rand = jax.random.normal(key0, (2,))
    >>> params = network.init(key1, x_rand)

    Using the MLP:

    >>> x = jnp.zeros(2)
    >>> y = network.apply(params, x)
    """

    feature_list: list
    nonlinearity: callable = nn.relu
    initial_scale: float = 1.0
    action_bias: Union[float, jnp.ndarray] = 0.0

    @nn.compact
    def __call__(self, x):
        # Define the forward pass
        for feature in self.feature_list[1:-1]:
            x = nn.Dense(
                feature,
                kernel_init=nn.initializers.variance_scaling(
                    self.initial_scale, mode="fan_avg", distribution="normal"
                ),
                bias_init=nn.initializers.zeros,
            )(x)
            x = self.nonlinearity(x)
        x = nn.Dense(
            self.feature_list[-1],
            kernel_init=nn.initializers.variance_scaling(
                self.initial_scale, mode="fan_avg", distribution="normal"
            ),
            bias_init=nn.initializers.zeros,
        )(x)
        return x + self.action_bias

    def initialize(self, key):
        """
        Initialize the model with random weights. Shorthand for `init`.
        :param key: random key
        :return: initial parameters
        """
        x_rand = jax.random.normal(key, (self.feature_list[0],))
        return self.init(key, x_rand)
    
class MLP_NORM(nn.Module):
    feature_list: list
    nonlinearity: callable = nn.relu
    initial_scale: float = 1.0
    action_bias: Union[float, jnp.ndarray] = 0.0
    out_dims_tanh: int = 0   # number of outputs to map with tanh
    out_dims_sigmoid: int = 0  # number of outputs to map with sigmoid

    @nn.compact
    def __call__(self, x):
        # Hidden layers
        for feature in self.feature_list[1:-1]:
            x = nn.Dense(
                feature,
                kernel_init=nn.initializers.variance_scaling(
                    self.initial_scale, mode="fan_avg", distribution="normal"
                ),
                bias_init=nn.initializers.zeros,
            )(x)
            x = self.nonlinearity(x)

        # Final layer
        final_out_dim = self.feature_list[-1]
        x = nn.Dense(
            final_out_dim,
            kernel_init=nn.initializers.variance_scaling(
                self.initial_scale, mode="fan_avg", distribution="normal"
            ),
            bias_init=nn.initializers.zeros,
        )(x)

        # Apply different activations to subsets of outputs
        outs = []
        idx = 0
        if self.out_dims_tanh > 0:
            outs.append(jnp.tanh(x[..., idx: idx + self.out_dims_tanh]))
            idx += self.out_dims_tanh
        if self.out_dims_sigmoid > 0:
            outs.append(nn.sigmoid(x[..., idx: idx + self.out_dims_sigmoid]))
            idx += self.out_dims_sigmoid
        if idx < final_out_dim:  # any leftover outputs remain linear
            outs.append(x[..., idx:])

        return jnp.concatenate(outs, axis=-1) + self.action_bias

    def initialize(self, key):
        x_rand = jax.random.normal(key, (self.feature_list[0],))
        return self.init(key, x_rand)


class OrthogonalMLP(MLP):
    @nn.compact
    def __call__(self, x):
        # Define the forward pass
        for feature in self.feature_list[1:-1]:
            x = nn.Dense(
                feature,
                kernel_init=nn.initializers.orthogonal(
                    scale=self.initial_scale
                ),
                bias_init=nn.initializers.zeros,
            )(x)
            x = self.nonlinearity(x)
        x = nn.Dense(
            self.feature_list[-1],
            kernel_init=nn.initializers.orthogonal(scale=self.initial_scale),
            bias_init=nn.initializers.zeros,
        )(x)
        return x + self.action_bias


class PiValue(NamedTuple):
    pi: distrax.Distribution
    value: jnp.ndarray


class ActorCriticPPO(MLP):
    initial_log_std: float = 0.0

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        # actor
        x = obs
        for feature in self.feature_list[1:-1]:
            x = nn.Dense(
                feature,
                kernel_init=nn.initializers.variance_scaling(
                    self.initial_scale, mode="fan_avg", distribution="normal"
                ),
                bias_init=nn.initializers.zeros,
            )(x)
            x = self.nonlinearity(x)
        x = nn.Dense(
            self.feature_list[-1],
            kernel_init=nn.initializers.variance_scaling(
                self.initial_scale, mode="fan_avg", distribution="normal"
            ),
            bias_init=nn.initializers.zeros,
        )(x)
        action_mean = x + self.action_bias
        # action_mean = nn.tanh(action_mean)

        action_logtstd = self.param(
            "log_std",
            nn.initializers.constant(self.initial_log_std),
            (self.feature_list[-1],),
        )
        action_std = jnp.maximum(jnp.exp(action_logtstd), 0.05)
        # create distribution object
        pi = distrax.MultivariateNormalDiag(action_mean, action_std)

        # critic
        x = obs
        for feature in self.feature_list[1:-1]:
            x = nn.Dense(
                feature,
                kernel_init=nn.initializers.variance_scaling(
                    self.initial_scale, mode="fan_avg", distribution="normal"
                ),
                bias_init=nn.initializers.zeros,
            )(x)
            x = self.nonlinearity(x)
        x = nn.Dense(
            1,
            kernel_init=nn.initializers.variance_scaling(
                self.initial_scale, mode="fan_avg", distribution="normal"
            ),
            bias_init=nn.initializers.zeros,
        )(x)
        value = jnp.squeeze(x, axis=-1)

        return PiValue(pi, value)
