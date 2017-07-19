# ----------------------------------------------------------------------
# Copyright (c) 2017, Jin-Man Park. All rights reserved.
# Contributors: Jin-Man Park and Jong-hwan Kim
# Affiliation: Robot Intelligence Technology Lab.(RITL), Korea Advanced Institute of Science and Technology (KAIST)
# URL: http://rit.kaist.ac.kr
# E-mail: jmpark@rit.kaist.ac.kr
# Citation: Jin-Man Park, and Jong-Hwan Kim. "Online recurrent extreme learning machine and its application to
# time-series prediction." Neural Networks (IJCNN), 2017 International Joint Conference on. IEEE, 2017.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
# ----------------------------------------------------------------------
# This code is originally from Numenta's Hierarchical Temporal Memory (HTM) code
# (Numenta Platform for Intelligent Computing (NuPIC))
# And modified to run Onine Recurrent Extreme Learning Machine (OR-ELM)
# ----------------------------------------------------------------------

import numpy as np


def NRMSE(data, pred):
  return np.sqrt(np.nanmean(np.square(pred-data)))/\
         np.nanstd(data)



def NRMSE_sliding(data, pred, windowSize):
  """
  Computing NRMSE in a sliding window
  :param data:
  :param pred:
  :param windowSize:
  :return: (window_center, NRMSE)
  """

  halfWindowSize = int(round(float(windowSize)/2))
  window_center = range(halfWindowSize, len(data)-halfWindowSize, int(round(float(halfWindowSize)/5.0)))
  nrmse = []
  for wc in window_center:
    nrmse.append(NRMSE(data[wc-halfWindowSize:wc+halfWindowSize],
                       pred[wc-halfWindowSize:wc+halfWindowSize]))

  return (window_center, nrmse)


def altMAPE(groundTruth, prediction):
  error = abs(groundTruth - prediction)
  altMAPE = 100.0 * np.sum(error) / np.sum(abs(groundTruth))
  return altMAPE


def MAPE(groundTruth, prediction):
  MAPE = np.nanmean(
    np.abs(groundTruth - prediction)) / np.nanmean(np.abs(groundTruth))
  return MAPE