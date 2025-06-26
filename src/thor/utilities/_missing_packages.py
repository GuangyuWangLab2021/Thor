import importlib
import logging
from functools import wraps


def require_packages(*package_names):
    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            missing_packages = []
            for package_name in package_names:
                try:
                    importlib.import_module(package_name)
                except ImportError:
                    missing_packages.append(package_name)

            if missing_packages:
                logger = logging.getLogger(__name__)
                logger.error("The following packages are required but not found:")
                for package_name in missing_packages:
                    logger.error(f"- {package_name}")
                logger.error("Please install the missing packages using 'pip install <package_name>'.")
                raise ImportError(f"Missing packages: {missing_packages}")

            return func(*args, **kwargs)

        return wrapper

    return decorator
