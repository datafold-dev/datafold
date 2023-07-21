#!/usr/bin/env python3

# by importing all objects from the three layers (with from X import *) this allows to access
# the objects in datafold directly
# e.g.
# import datafold
# datafold.DiffusionMaps()

from datafold._version import Version
from datafold.appfold.edmd import *  # noqa F403
from datafold.dynfold import *  # noqa F403
from datafold.pcfold import *  # noqa F403

__version__ = Version.v_short
