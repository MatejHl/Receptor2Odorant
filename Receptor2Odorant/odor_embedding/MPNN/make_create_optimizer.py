import optax

from Receptor2Odorant.utils import TrainState_with_epoch_and_rngs

def make_create_optimizer(model, option = 'adamw_polynomial', **kwargs): 
    if option == 'adamw_polynomial':
        transition_steps = kwargs.get('transition_steps')
        def create_optimizer(params, rngs = None, learning_rate=0.01):
            scheduler = optax.polynomial_schedule(init_value = learning_rate, 
                                                    end_value = 5e-4, 
                                                    power = 0.8,
                                                    transition_steps = transition_steps, 
                                                    transition_begin=0)

            opt = optax.chain(optax.adamw(learning_rate = scheduler))
            state = TrainState_with_epoch_and_rngs.create(apply_fn = model.apply, 
                                            params = params,
                                            tx = opt,
                                            rngs = rngs,
                                            )
            return state, scheduler
    return create_optimizer