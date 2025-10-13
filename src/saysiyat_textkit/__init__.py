from .normalization import *
from .segmentation import *
from .tagging import *
from .pipeline import *

from importlib.metadata import version, PackageNotFoundError  # py3.8+

def _dist_version():
    # Use your distribution name as declared in pyproject [project].name
    for dist_name in ("saysiyat_textkit", "saysiyat-textkit"):
        try:
            return version(dist_name)
        except PackageNotFoundError:
            pass
    return "0+unknown"

__version__ = _dist_version()