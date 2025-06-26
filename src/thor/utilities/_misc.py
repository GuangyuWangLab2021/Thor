import inspect

def get_kwargs(func):
    signature = inspect.signature(func)

    keyword_args = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    return keyword_args


def update_kwargs_exclusive(func, kwargs_dict_for_update):
    """
    Update kwargs of a function with a dictionary, excluding non-existent kwargs
    """

    kwargs = get_kwargs(func)
    kwargs = {kw: kwargs_dict_for_update.get(kw, kwargs[kw]) for kw in kwargs}

    return kwargs

def update_kwargs(func, kwargs_dict_for_update):
    """
    Update kwargs of a function with a dictionary, including non-existent kwargs
    """

    kwargs = get_kwargs(func)
    updated_kwargs = kwargs.copy()

    for kw, value in kwargs_dict_for_update.items():
        if kw in kwargs:
            updated_kwargs[kw] = value
        else:
            updated_kwargs[kw] = value

    return updated_kwargs
