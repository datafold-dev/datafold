""" Diffusion maps module.
"""

from datafold.dynfold.diffusion_maps import *
from datafold.dynfold.dmd import *
from datafold.dynfold.gpr import *
from datafold.dynfold.outofsample import *
from datafold.dynfold.plot import *

try:
    # TODO: possibly remove interactive plotting
    from datafold.dynfold.plot_interactive import *
except ImportError:
    pass  # Do nothing -- this is most likely because Ipython imports failed
