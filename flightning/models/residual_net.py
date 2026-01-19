import flax.linen as nnj
import jax.numpy as jnp
import torch
import torch.nn as nn
import jax
from flax.core import freeze, unfreeze

class ResidualNetFlax(nnj.Module):
    hidden: int = 256
    out_size: int = 6

    @nnj.compact
    def __call__(self, x):
        x = nnj.Dense(self.hidden)(x)
        x = nnj.relu(x)
        x = nnj.LayerNorm()(x)
        x = nnj.Dense(self.hidden)(x)
        x = nnj.relu(x)
        x = nnj.LayerNorm()(x)
        x = nnj.Dense(self.hidden)(x)
        x = nnj.relu(x)
        x = nnj.LayerNorm()(x)
        x = nnj.Dense(self.hidden)(x)
        x = nnj.relu(x)
        x = nnj.Dense(self.out_size)(x)
        return x



class ResidualNetTorch(nn.Module):
    def __init__(self, in_size=19, hidden=256, out_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_size),
        )

    def forward(self, x):
        return self.net(x)


# #Initialize flax model
# flax_model = ResidualNetFlax(hidden=256, out_size=3)
# rng = jax.random.key(0)
# dummy_x = jnp.ones((1, 19))
# params = flax_model.init(rng, dummy_x)


# #Initialize torch model
# torch_model = ResidualNetTorch()
# full_path = '/home/yanrui/tempstorage4/rpg_flightning/data/model_3.pt'
# ckpt = torch.load(full_path, map_location="cpu")
# state_dict = ckpt["model_state"] 
# torch_model.load_state_dict(state_dict)
# # state_dict = torch.load(full_path, map_location=torch.device('cpu'))
# for k, v in state_dict.items():
#     print(k, v.shape)




# params_flax = unfreeze(params)

# # Layer 0: Linear(in=20, hidden=256)
# params_flax['params']['Dense_0']['kernel'] = jnp.array(state_dict['net.0.weight'].T.numpy())
# params_flax['params']['Dense_0']['bias']   = jnp.array(state_dict['net.0.bias'].numpy())

# # LayerNorm
# params_flax['params']['LayerNorm_0']['scale'] = jnp.array(state_dict['net.2.weight'].numpy())
# params_flax['params']['LayerNorm_0']['bias']  = jnp.array(state_dict['net.2.bias'].numpy())

# # Layer 1: Linear(hidden=256, hidden=256)
# params_flax['params']['Dense_1']['kernel'] = jnp.array(state_dict['net.3.weight'].T.numpy())
# params_flax['params']['Dense_1']['bias']   = jnp.array(state_dict['net.3.bias'].numpy())

# # Layer 2: Linear(hidden=256, out=6)
# params_flax['params']['Dense_2']['kernel'] = jnp.array(state_dict['net.5.weight'].T.numpy())
# params_flax['params']['Dense_2']['bias']   = jnp.array(state_dict['net.5.bias'].numpy())

# params_flax = freeze(params_flax)


# import numpy as np

# # Torch forward
# x_torch = torch.randn(1, 19)
# x_torch = torch.asarray([ 0.5980,  0.8787, -0.7539,  0.0337, -1.2266,  0.2220,  0.3696, -0.2098,
#          -0.2836,  0.3290, -0.9231,  0.5901,  0.5150, -0.4487,  1.2988,  1.9111,
#          -0.0085,  1.4803, -0.2838])
# y_torch = torch_model(x_torch).detach().numpy()

# # Flax forward
# x_jax = jnp.array(x_torch.numpy())
# y_flax = flax_model.apply(params_flax, x_jax)

# print("Torch:", y_torch)
# print("Flax:", np.array(y_flax))
# print("Max diff:", np.abs(y_torch - np.array(y_flax)).max())
