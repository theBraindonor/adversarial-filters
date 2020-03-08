#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Pull the filters together to be easily included in the experiment and evaluation code.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"

from filter.GaussianFilter import *
from filter.GaussianNoiseFilter import *
from filter.ColorBlindnessFilter import *
from filter.DaltonizationFilter import *
from filter.HistogramEqualizationFilter import *
from filter.FourierGaussianFilter import *
from filter.FourierEllipsoidFilter import *
from filter.FourierUniformFilter import *
from filter.WaveletDenoiseFilter import *
from filter.SpeckleNoiseFilter import *
from filter.SaltPepperNoiseFilter import *
from filter.DaltonizationLabEnchanceFilter import *
