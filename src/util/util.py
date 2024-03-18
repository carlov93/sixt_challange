import time
import logging
from functools import wraps
from typing import Any, Dict


logger = logging.getLogger("bike_prediction")


def timed(func):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        msg = f"""Finished function {func.__name__} successfully in {round(end - start, 2)} seconds"""
        logger.info(msg)
        return result

    return wrapper


def logging_setup(config: Dict):
    """
    setup logging based on the configuration

    :param config: the parsed config tree
    """
    log_conf = config["logging"]

    if log_conf["enabled"]:
        level = logging._nameToLevel[log_conf["level"].upper()]
    else:
        level = logging.NOTSET

    logger.setLevel(level)
    

def set_pipeline_step_log_level(level: str = "INFO", logging_format: str = None):
    if not logging_format:
        logging_format = "%(asctime)s - %(threadName)s - "
        logging_format += "%(name)s:%(lineno)d - %(levelname)s - %(message)s"
    logging.basicConfig(format=logging_format, level=level)


def pipeline_logging_config(_func=None, *, level: str = "INFO", logging_format: str = None):
    def decorator(func, *args, **kwargs):
        @wraps(func)
        def wrapper(*args, **kwargs):
            set_pipeline_step_log_level(level=level, logging_format=logging_format)
            result = func(*args, **kwargs)
            return result

        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)