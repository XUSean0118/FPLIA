import copy

models = {}

def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False):
    model_args = copy.deepcopy(model_spec.get('args', {}))
    if args is not None:
        model_args.update(args)

    model = models[model_spec['name']](**model_args)

    if load_sd:
        model.load_state_dict(model_spec['sd'])
        
    return model
