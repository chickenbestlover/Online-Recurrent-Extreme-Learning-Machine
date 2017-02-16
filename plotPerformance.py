#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------


from matplotlib import pyplot as plt
from errorMetrics import *
import pandas as pd

from pylab import rcParams
from plot import ExperimentResult, plotAccuracy, computeSquareDeviation, computeLikelihood, plotLSTMresult
from nupic.encoders.scalar import ScalarEncoder as NupicScalarEncoder

rcParams.update({'figure.autolayout': True})
rcParams.update({'figure.facecolor': 'white'})
rcParams.update({'ytick.labelsize': 8})
rcParams.update({'figure.figsize': (12, 6)})
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
plt.ion()
plt.close('all')

window = 100
skipTrain = 500
figPath = './result/'

def getDatetimeAxis():
  """
  use datetime as x-axis
  """
  dataSet = 'nyc_taxi'
  filePath = './data/' + dataSet + '.csv'
  data = pd.read_csv(filePath, header=0, skiprows=[1, 2],
                     names=['datetime', 'value', 'timeofday', 'dayofweek'])

  xaxisDate = pd.to_datetime(data['datetime'])
  return xaxisDate


def computeAltMAPE(truth, prediction, startFrom=0):
  return np.nanmean(np.abs(truth[startFrom:] - prediction[startFrom:]))/np.nanmean(np.abs(truth[startFrom:]))


def computeNRMSE(truth, prediction, startFrom=0):
  squareDeviation = computeSquareDeviation(prediction, truth)
  squareDeviation[:startFrom] = None
  return np.sqrt(np.nanmean(squareDeviation))/np.nanstd(truth)


def loadExperimentResult(filePath):
  expResult = pd.read_csv(filePath, header=0, skiprows=[1, 2],
                            names=['step', 'value', 'prediction5'])
  groundTruth = np.roll(expResult['value'], -5)
  prediction5step = np.array(expResult['prediction5'])
  return (groundTruth, prediction5step)



if __name__ == "__main__":
  xaxisDate = getDatetimeAxis()
  plt.figure(1)
  dataSet = 'nyc_taxi'
  filePath = './prediction/' + dataSet + '_TM_pred.csv'
  (tmTruth, tmPrediction) = loadExperimentResult('./prediction/' + dataSet + '_TM_pred.csv')
  squareDeviation = computeSquareDeviation(tmPrediction, tmTruth)
  squareDeviation[:skipTrain] = None
  nrmseTM = plotAccuracy((squareDeviation, xaxisDate),
                         tmTruth,
                         window=window,
                         errorType='square_deviation',
                         label='TM')

  '''
  (elmD75Truth, elmD75Prediction) = loadExperimentResult('./prediction/' + dataSet + '_elmDecay75_pred.csv')
  squareDeviation = computeSquareDeviation(elmD75Prediction, elmD75Truth)
  squareDeviation[:skipTrain] = None
  nrmseELM = plotAccuracy((squareDeviation, xaxisDate),
                          elmD75Truth,
                          window=window,
                          errorType='square_deviation',
                          label='ELM with decaying factor #HN=75')


  (elmD50Truth, elmD50Prediction) = loadExperimentResult('./prediction/' + dataSet + '_elmDecay50_pred.csv')
  squareDeviation = computeSquareDeviation(elmD50Prediction, elmD50Truth)
  squareDeviation[:skipTrain] = None
  nrmseELM = plotAccuracy((squareDeviation, xaxisDate),
                          elmD50Truth,
                          window=window,
                          errorType='square_deviation',
                          label='ELM with decaying factor #HN=50')

  (elmD25Truth, elmD25Prediction) = loadExperimentResult('./prediction/' + dataSet + '_elmDecay25_pred.csv')
  squareDeviation = computeSquareDeviation(elmD25Prediction, elmD25Truth)
  squareDeviation[:skipTrain] = None
  nrmseELM = plotAccuracy((squareDeviation, xaxisDate),
                          elmD25Truth,
                          window=window,
                          errorType='square_deviation',
                          label='ELM with decaying factor #HN=25')

  (elmD10Truth, elmD10Prediction) = loadExperimentResult('./prediction/' + dataSet + '_elmDecay10_pred.csv')
  squareDeviation = computeSquareDeviation(elmD10Prediction, elmD10Truth)
  squareDeviation[:skipTrain] = None
  nrmseELM = plotAccuracy((squareDeviation, xaxisDate),
                          elmD10Truth,
                          window=window,
                          errorType='square_deviation',
                          label='ELM with decaying factor #HN=10')

  (elmD5Truth, elmD5Prediction) = loadExperimentResult('./prediction/' + dataSet + '_elmDecay5_pred.csv')
  squareDeviation = computeSquareDeviation(elmD5Prediction, elmD5Truth)
  squareDeviation[:skipTrain] = None
  nrmseELM = plotAccuracy((squareDeviation, xaxisDate),
                          elmD5Truth,
                          window=window,
                          errorType='square_deviation',
                          label='ELM with decaying factor #HN=5')



  (elm5Truth, elm5Prediction) = loadExperimentResult('./prediction/' + dataSet + '_elm5_pred.csv')
  squareDeviation = computeSquareDeviation(elm5Prediction, elm5Truth)
  squareDeviation[:skipTrain] = None
  nrmseELM = plotAccuracy((squareDeviation, xaxisDate),
                          elm5Truth,
                          window=window,
                          errorType='square_deviation',
                          label='FELM_#HN=5')

  (elm10Truth, elm10Prediction) = loadExperimentResult('./prediction/' + dataSet + '_elm10_pred.csv')
  squareDeviation = computeSquareDeviation(elm10Prediction, elm10Truth)
  squareDeviation[:skipTrain] = None
  nrmseELM = plotAccuracy((squareDeviation, xaxisDate),
                          elm10Truth,
                          window=window,
                          errorType='square_deviation',
                          label='FELM_#HN=10')

  (elm25Truth, elm25Prediction) = loadExperimentResult('./prediction/' + dataSet + '_elm25_pred.csv')
  squareDeviation = computeSquareDeviation(elm25Prediction, elm25Truth)
  squareDeviation[:skipTrain] = None
  nrmseELM = plotAccuracy((squareDeviation, xaxisDate),
                          elm25Truth,
                          window=window,
                          errorType='square_deviation',
                          label='FELM_#HN=25')

  (elm50Truth, elm50Prediction) = loadExperimentResult('./prediction/' + dataSet + '_elm50_pred.csv')
  squareDeviation = computeSquareDeviation(elm50Prediction, elm50Truth)
  squareDeviation[:skipTrain] = None
  nrmseELM = plotAccuracy((squareDeviation, xaxisDate),
                          elm50Truth,
                          window=window,
                          errorType='square_deviation',
                          label='FELM_#HN=50')

  (elm100Truth, elm100Prediction) = loadExperimentResult('./prediction/' + dataSet + '_elm100_pred.csv')
  squareDeviation = computeSquareDeviation(elm100Prediction, elm100Truth)
  squareDeviation[:skipTrain] = None
  nrmseELM = plotAccuracy((squareDeviation, xaxisDate),
                          elm100Truth,
                          window=window,
                          errorType='square_deviation',
                          label='FELM_#HN=100')
  (elm500Truth, elm500Prediction) = loadExperimentResult('./prediction/' + dataSet + '_elm500_pred.csv')
  squareDeviation = computeSquareDeviation(elm500Prediction, elm500Truth)
  squareDeviation[:skipTrain] = None
  nrmseELM = plotAccuracy((squareDeviation, xaxisDate),
                          elm500Truth,
                          window=window,
                          errorType='square_deviation',
                          label='FELM_#HN=500')
  (elm1000Truth, elm1000Prediction) = loadExperimentResult('./prediction/' + dataSet + '_elm1000_pred.csv')
  squareDeviation = computeSquareDeviation(elm100Prediction, elm1000Truth)
  squareDeviation[:skipTrain] = None
  nrmseELM = plotAccuracy((squareDeviation, xaxisDate),
                          elm1000Truth,
                          window=window,
                          errorType='square_deviation',
                          label='FELM_#HN=1000')
  '''

  (elmTruth, elmPrediction) = loadExperimentResult('./prediction/' + dataSet + '_elmDecay+1_pred.csv')
  squareDeviation = computeSquareDeviation(elmPrediction, elmTruth)
  squareDeviation[:skipTrain] = None
  nrmseELM = plotAccuracy((squareDeviation, xaxisDate),
                          elmTruth,
                          window=window,
                          errorType='square_deviation',
                          label='ELMDecay_#HN=25')

  (elmABTruth, elmABPrediction) = loadExperimentResult('./prediction/' + dataSet + '_elmAB23_pred.csv')
  squareDeviation = computeSquareDeviation(elmABPrediction, elmABTruth)
  squareDeviation[:skipTrain] = None
  nrmseELM = plotAccuracy((squareDeviation, xaxisDate),
                          elmABTruth,
                          window=window,
                          errorType='square_deviation',
                          label='ELMAB_#HN=23')

  plt.legend()
  plt.savefig(figPath + 'continuous.pdf')

  startFrom = skipTrain

  altMAPETM = computeAltMAPE(tmTruth, tmPrediction, startFrom)
  altMAPEELM = computeAltMAPE(elmTruth, elmPrediction, startFrom)
  truth = tmTruth
  nrmseTMmean = np.sqrt(np.nanmean(nrmseTM)) / np.nanstd(truth)
  nrmseELMmean = np.sqrt(np.nanmean(nrmseELM)) / np.nanstd(truth)

  fig, ax = plt.subplots(nrows=1, ncols=2)
  inds = np.arange(2)
  ax1 = ax[0]
  width = 0.5
  ax1.bar(inds, [
                 nrmseELMmean,
                 nrmseTMmean], width=width)
  ax1.set_xticks(inds+width/2)
  ax1.set_ylabel('NRMSE')
  ax1.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
  ax1.set_xticklabels( ('ELM', 'HTM') )
  for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_rotation('vertical')

  ax3 = ax[1]
  ax3.bar(inds, [
                 altMAPEELM,
                 altMAPETM], width=width, color='b')
  ax3.set_xticks(inds+width/2)
  ax3.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
  ax3.set_ylabel('MAPE')
  ax3.set_xticklabels( ('ELM', 'HTM') )
  for tick in ax3.xaxis.get_major_ticks():
    tick.label.set_rotation('vertical')

  plt.show()

  plt.savefig(figPath + 'model_performance_summary_neural_networks.pdf')



  raw_input()