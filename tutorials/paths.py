#!/usr/bin/env python3

import os
import sys

tutorial_path = os.path.dirname(os.path.realpath(__file__))
datafold_path = os.path.abspath(os.path.join(tutorial_path, os.pardir))


def add_paths_to_sys():
    sys.path.insert(0, tutorial_path)
    sys.path.insert(0, datafold_path)
