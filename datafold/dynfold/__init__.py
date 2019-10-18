""" Diffusion maps module.

"""

# NOTE: leave out leading zeros in day or month, because this aligns with how it is displayed in `pip list`
# Examples: "2000.1.1", "2000.6.30", "2000.12.31"
__version__ = "2019.6.6"  # YYYY.[1]M.[2-3]D

from datafold.dynfold.diffusion_maps import *
from datafold.dynfold.geometric_harmonics import *
from datafold.dynfold.koopman import *
from datafold.dynfold.plot import *
from datafold.dynfold.utils import downsample  # TODO: replace by pcfold downsample!

try:
    from datafold.dynfold.plot_interactive import *  # TODO: possibly remove interactive plotting
except ImportError:
    pass  # Do nothing -- this is most likely because Ipython imports failed
