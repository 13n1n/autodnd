import functools
import logging


def log_call(fn):
    @functools.wraps(fn)
    def __wrapped(*args, **kwargs):
        logging.info(f"Calling {fn.__name__} {args} {kwargs}")
        return fn(*args, **kwargs)
    return __wrapped
