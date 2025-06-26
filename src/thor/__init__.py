try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

name = "Thor"
package_name = "thor"
__version__ = importlib_metadata.version(package_name)

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# nice logging outputs
from rich.console import Console
from rich.logging import RichHandler
console = Console(force_terminal=True, width=150)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=True)
formatter = logging.Formatter("Thor: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# this prevents double outputs
logger.propagate = False


from . import pl, utils, analy, pp, VAE
from .finest import fineST
