import torch
import torch.nn as nn
import flax
from flax.training.train_state import TrainState
# Example torch model
torch_model = nn.Linear(15, 4)   # same as flax.Dense(4)
# assume it's trained
state_dict = torch_model.state_dict()


import jax.numpy as jnp

# extract and convert
weight = state_dict["weight"].cpu().numpy()   # (4, 15)
bias = state_dict["bias"].cpu().numpy()       # (4,)

# transpose weight for Flax
kernel = jnp.array(weight.T)  # (15, 4)
bias = jnp.array(bias)        # (4,)


# unpack flax params
params_flax = train_state.params.unfreeze()

# update weights
params_flax["params"]["kernel"] = kernel
params_flax["params"]["bias"] = bias

# re-freeze
params_flax = flax.core.freeze(params_flax)

# new train_state with torch weights
train_state = train_state.replace(params=params_flax)


x_torch = torch.randn(1, 15)
y_torch = torch_model(x_torch)

x_jax = jnp.array(x_torch.numpy())
y_jax = train_state.apply_fn(train_state.params, x_jax)

print(y_torch.detach().numpy())
print(np.array(y_jax))
