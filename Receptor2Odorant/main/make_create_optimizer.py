import optax

from Receptor2Odorant.utils import TrainState_with_epoch_and_rngs
from Receptor2Odorant.schedulers import transformer_schedule

def make_create_optimizer(model, option = 'adamw_polynomial', **kwargs):
    if option == 'adamw_polynomial':
        transition_steps = kwargs.get('transition_steps')
        def create_optimizer(params, rngs = None, learning_rate=0.01):
            scheduler = optax.polynomial_schedule(init_value = learning_rate, 
                                                    end_value = 1e-5, 
                                                    power = 1.0, 
                                                    transition_steps = transition_steps,
                                                    transition_begin=0)
            opt = optax.chain(optax.adamw(learning_rate = scheduler))
            # NOTE: This handles updates of opt_state and params and init of opt
            state = TrainState_with_epoch_and_rngs.create(apply_fn = model.apply, 
                                            params = params,
                                            tx = opt,
                                            rngs = rngs,
                                            )
            return state, scheduler
    elif option == 'adam_transformer':
        warmup_steps = kwargs.get('warmup_steps')
        def create_optimizer(params, rngs = None, learning_rate=0.01):
            scheduler = transformer_schedule(init_value = learning_rate, 
                                                warmup_steps = warmup_steps)
            opt = optax.chain(optax.adam(learning_rate = scheduler))
            state = TrainState_with_epoch_and_rngs.create(apply_fn = model.apply, 
                                            params = params,
                                            tx = opt,
                                            rngs = rngs,
                                            )
            return state, scheduler
    return create_optimizer