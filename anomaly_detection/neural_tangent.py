from jax import random
from jax.experimental import stax

init_fn, apply_fn = stax.serial(
    stax.Dense(512), stax.Relu,
    stax.Dense(512), stax.Relu,
    stax.Dense(1)
)

key = random.PRNGKey(1)
x = random.normal(key, (10, 100))
_, params = init_fn(key, input_shape=x.shape)

y = apply_fn(params, x)  # (10, 1) np.ndarray outputs of the neural network