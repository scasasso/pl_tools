# -*- coding: utf-8 -*-
################################################################################
#
# File:     skmodel.py
#
# Product:  Predictive Layer Genius Classif Library
# Author:   Momo  
# Date:     03 April 2015
#
# Scope:    The file contains the representation of scikit learn model.
#
# Copyright (c) 2015, Predictive Layer Limited.  All Rights Reserved.
#
# The contents of this software are proprietary and confidential to the author.
# No part of this program may be photocopied,  reproduced, or translated into
# another programming language without prior written consent of the author.
#
#
# $Id$
#
################################################################################
import json
import numpy
import pickle
from sklearn.externals import joblib
from sklearn.base import clone as skclone
from plmodel import PLModel


class SKModel(PLModel):
    
    def __init__(self, model):
        PLModel.__init__(self, model)
        return
