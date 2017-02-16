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
import datetime
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
  plt.figure(figsize=(8,4))
  dataSet = 'nyc_taxi'




  (elmTruth, elmPrediction) = loadExperimentResult('./prediction/' + dataSet + '_Decay0.9WAOSELM10_pred.csv')
  squareDeviation = computeSquareDeviation(elmPrediction, elmTruth)
  squareDeviation[:skipTrain] = None
  nrmseBasicELM10 = plotAccuracy((squareDeviation, xaxisDate),
                          elmTruth,
                          window=window,
                          errorType='square_deviation',
                          label='WAOSELM_#HN=10')
  altMAPEELM10 = computeAltMAPE(elmTruth, elmPrediction, skipTrain)

  (elmTruth, elmPrediction) = loadExperimentResult('./prediction/' + dataSet + '_Decay0.9WAOSELM20_pred.csv')
  squareDeviation = computeSquareDeviation(elmPrediction, elmTruth)
  squareDeviation[:skipTrain] = None
  nrmseBasicELM20 = plotAccuracy((squareDeviation, xaxisDate),
                          elmTruth,
                          window=window,
                          errorType='square_deviation',
                          label='WAOSELM_#HN=20')
  altMAPEELM20 = computeAltMAPE(elmTruth, elmPrediction, skipTrain)

  (elmTruth, elmPrediction) = loadExperimentResult('./prediction/' + dataSet + '_Decay0.9WAOSELM25_pred.csv')
  squareDeviation = computeSquareDeviation(elmPrediction, elmTruth)
  squareDeviation[:skipTrain] = None
  nrmseBasicELM25 = plotAccuracy((squareDeviation, xaxisDate),
                          elmTruth,
                          window=window,
                          errorType='square_deviation',
                          label='WAOSELM_#HN=25')
  altMAPEELM25 = computeAltMAPE(elmTruth, elmPrediction, skipTrain)

  (elmTruth, elmPrediction) = loadExperimentResult('./prediction/' + dataSet + '_Decay0.9WAOSELM50_pred.csv')
  squareDeviation = computeSquareDeviation(elmPrediction, elmTruth)
  squareDeviation[:skipTrain] = None
  nrmseBasicELM50 = plotAccuracy((squareDeviation, xaxisDate),
                          elmTruth,
                          window=window,
                          errorType='square_deviation',
                          label='WAOSELM_#HN=50')
  altMAPEELM50 = computeAltMAPE(elmTruth, elmPrediction, skipTrain)

  (elmTruth, elmPrediction) = loadExperimentResult('./prediction/' + dataSet + '_Decay0.9WAOSELM100_pred.csv')
  squareDeviation = computeSquareDeviation(elmPrediction, elmTruth)
  squareDeviation[:skipTrain] = None
  nrmseBasicELM100 = plotAccuracy((squareDeviation, xaxisDate),
                          elmTruth,
                          window=window,
                          errorType='square_deviation',
                          label='WAOSELM_#HN=100')
  altMAPEELM100 = computeAltMAPE(elmTruth, elmPrediction, skipTrain)
  '''
  (elmTruth, elmPrediction) = loadExperimentResult('./prediction/' + dataSet + '_Decay0.9WAOSELM200_pred.csv')
  squareDeviation = computeSquareDeviation(elmPrediction, elmTruth)
  squareDeviation[:skipTrain] = None
  nrmseBasicELM200 = plotAccuracy((squareDeviation, xaxisDate),
                          elmTruth,
                          window=window,
                          errorType='square_deviation',
                          label='WAOSELM_#HN=200')
  altMAPEELM200 = computeAltMAPE(elmTruth, elmPrediction, skipTrain)

  (elmTruth, elmPrediction) = loadExperimentResult('./prediction/' + dataSet + '_Decay0.9WAOSELM400_pred.csv')
  squareDeviation = computeSquareDeviation(elmPrediction, elmTruth)
  squareDeviation[:skipTrain] = None
  nrmseBasicELM400 = plotAccuracy((squareDeviation, xaxisDate),
                          elmTruth,
                          window=window,
                          errorType='square_deviation',
                          label='WAOSELM_#HN=400')
  '''
  altMAPEELM200 = 1000
  altMAPEELM400 = 1000

  altMAPEELM800 = 1000
  altMAPEELM1600 = 1000

  #plt.xlim([datetime.datetime(2014,7,13,12,0),datetime.datetime(2014,7,15,0,0)])
  plt.xlim([datetime.datetime(2014, 7, 13, 12, 0), datetime.datetime(2014, 9, 12, 0, 0)])

  plt.ylim([0.0,1.5])
  #plt.ylim([0.2,0.9])
  #plt.legend(loc='lower right')
  plt.legend(loc='lower right')
  plt.legend()
  plt.savefig(figPath + 'squrare_deviation_WADOSELM0.900.pdf')

  startFrom = skipTrain

  altMAPEELM = computeAltMAPE(elmTruth, elmPrediction, startFrom)

  truth = elmTruth
  nrmseBasicELM10mean = np.sqrt(np.nanmean(nrmseBasicELM10)) / np.nanstd(truth)
  nrmseBasicELM20mean = np.sqrt(np.nanmean(nrmseBasicELM20)) / np.nanstd(truth)
  nrmseBasicELM25mean = np.sqrt(np.nanmean(nrmseBasicELM25)) / np.nanstd(truth)
  nrmseBasicELM50mean = np.sqrt(np.nanmean(nrmseBasicELM50)) / np.nanstd(truth)
  nrmseBasicELM100mean = np.sqrt(np.nanmean(nrmseBasicELM100)) / np.nanstd(truth)
  nrmseBasicELM200mean = 1000
  nrmseBasicELM400mean = 1000
  nrmseBasicELM800mean = 1000
  nrmseBasicELM1600mean = 1000

  fig, ax = plt.subplots(nrows=1, ncols=2)
  inds = np.arange(9)
  ax1 = ax[0]
  width = 0.5
  ax1.bar(inds, [
      nrmseBasicELM10mean,
      nrmseBasicELM20mean,
      nrmseBasicELM25mean,
      nrmseBasicELM50mean,
      nrmseBasicELM100mean,
      nrmseBasicELM200mean,
      nrmseBasicELM400mean,
      nrmseBasicELM800mean,
      nrmseBasicELM1600mean,
  ], width=width)
  ax1.set_xticks(inds+width/2)
  ax1.set_ylabel('NRMSE')
  ax1.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
  ax1.set_xticklabels( ('WADOSELM10','WADOSELM20','WADOSELM25', 'WADOSELM50', 'WADOSELM100', 'WADOSELM200', 'WADOSELM400', 'WADOSELM800','WADOSELM1600' ) )
  for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_rotation('vertical')

  ax1.set_ylim([0,2])

  ax3 = ax[1]
  ax3.bar(inds, [
      altMAPEELM10,
      altMAPEELM20,
      altMAPEELM25,
      altMAPEELM50,
      altMAPEELM100,
      altMAPEELM200,
      altMAPEELM400,
      altMAPEELM800,
      altMAPEELM1600,

  ], width=width, color='b')
  ax3.set_xticks(inds+width/2)
  ax3.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
  ax3.set_ylabel('MAPE')
  ax3.set_xticklabels(('WAODSELM10','WAODSELM20','WADOSELM25', 'WADOSELM50', 'WADOSELM100', 'WADOSELM200', 'WADOSELM400','WADOSELM800', 'WADOSELM1600' ) )
  for tick in ax3.xaxis.get_major_ticks():
    tick.label.set_rotation('vertical')

  ax3.set_ylim([0,0.5])
  plt.show()
  print (
      altMAPEELM10,
      altMAPEELM20,
      altMAPEELM25,
      altMAPEELM50,
      altMAPEELM100,
      altMAPEELM200,
      altMAPEELM400,
     # altMAPEELM800,
     # altMAPEELM1600
                 )
  plt.savefig(figPath + 'model_performance_summary_WADOSELM0.900.pdf')



  #raw_input()