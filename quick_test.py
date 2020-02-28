#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet')
model.summary()
