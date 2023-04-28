from optax._src import base
from jax import numpy as jnp


def transformer_schedule(init_value: float, warmup_steps: int) -> base.Schedule:
    """
    scheduler suggested in Transformer paper.
    """
    if not warmup_steps > 0:
      raise ValueError('The transformer_schedule requires positive warmup_steps!')

    def schedule(count):
        arg1 = jnp.reciprocal(jnp.sqrt(count))
        arg2 = count * (warmup_steps ** -1.5)
        return init_value * jnp.minimum(arg1, arg2)

    return schedule