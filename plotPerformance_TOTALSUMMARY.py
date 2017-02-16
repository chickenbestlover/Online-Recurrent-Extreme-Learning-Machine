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




  (elmTruth, elmPrediction) = loadExperimentResult('./prediction/' + dataSet + '_RecurrentELM_Decay01_Whitened_23_pred.csv')
  squareDeviation = computeSquareDeviation(elmPrediction, elmTruth)
  squareDeviation[:skipTrain] = None
  nrmseBasicELM23 = plotAccuracy((squareDeviation, xaxisDate),
                          elmTruth,
                          window=window,
                          errorType='square_deviation',
                          label='RecurrentELM_#HN=23')
  altMAPEELM23 = computeAltMAPE(elmTruth, elmPrediction, skipTrain)

  (tmTruth, tmPrediction) = loadExperimentResult('./prediction/' + dataSet + '_TM_pred.csv')

  squareDeviation = computeSquareDeviation(tmPrediction, tmTruth)
  squareDeviation[:skipTrain] = None

  nrmseTM = plotAccuracy((squareDeviation, xaxisDate),
                         tmTruth,
                         window=window,
                         errorType='square_deviation',
                         label='TM')
  altMAPETM = computeAltMAPE(tmTruth, tmPrediction, skipTrain)

  (lstmTruth, lstmPrediction) = loadExperimentResult('./prediction/' + dataSet + '_lstm_pred.csv')

  squareDeviation = computeSquareDeviation(tmPrediction, tmTruth)
  squareDeviation[:skipTrain] = None

  nrmseTM = plotAccuracy((squareDeviation, xaxisDate),
                         tmTruth,
                         window=window,
                         errorType='square_deviation',
                         label='LSTM')
  altMAPELSTM = computeAltMAPE(lstmTruth, lstmPrediction, skipTrain)

  #plt.xlim([datetime.datetime(2014,7,13,12,0),datetime.datetime(2014,7,15,0,0)])
  plt.xlim([datetime.datetime(2014, 7, 13, 12, 0), datetime.datetime(2014, 9, 12, 0, 0)])

  plt.ylim([0.0,1.5])
  #plt.ylim([0.2,0.9])
  #plt.legend(loc='lower right')
  plt.legend(loc='lower right')
  plt.legend()
  plt.savefig(figPath + 'squrare_deviation_TOTAL.pdf')

  startFrom = skipTrain

  altMAPEELM = computeAltMAPE(elmTruth, elmPrediction, startFrom)

  truth = elmTruth
  nrmseBasicELM23mean = np.sqrt(np.nanmean(nrmseBasicELM23)) / np.nanstd(truth)
  nrmseTMmean= np.sqrt(np.nanmean(nrmseTM)) / np.nanstd(truth)
  nrmseLSTMmean= np.sqrt(np.nanmean(nrmseTM)) / np.nanstd(truth)

 # nrmseBasicELM100mean = np.sqrt(np.nanmean(nrmseBasicELM100)) / np.nanstd(truth)
 # nrmseBasicELM200mean = np.sqrt(np.nanmean(nrmseBasicELM200)) / np.nanstd(truth)
 # nrmseBasicELM400mean = np.sqrt(np.nanmean(nrmseBasicELM400)) / np.nanstd(truth)
 # nrmseBasicELM800mean = np.sqrt(np.nanmean(nrmseBasicELM800)) / np.nanstd(truth)

  fig, ax = plt.subplots(nrows=1, ncols=2)
  inds = np.arange(3)
  ax1 = ax[0]
  width = 0.5
  ax1.bar(inds, [
      nrmseBasicELM23mean,
      nrmseTMmean,
      nrmseLSTMmean
  ], width=width)
  ax1.set_xticks(inds+width/2)
  ax1.set_ylabel('NRMSE')
  ax1.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
  ax1.set_xticklabels( ('RecurrentELM23','TM','LSTM','RecurrentELM25', 'RecurrentELM50', 'RecurrentELM100', 'RecurrentELM200', 'RecurrentELM400', 'RecurrentELM800' ) )
  for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_rotation('vertical')

  ax3 = ax[1]
  ax3.bar(inds, [
      altMAPEELM23,
      altMAPETM,
      altMAPELSTM,
                 ], width=width, color='b')
  ax3.set_xticks(inds+width/2)
  ax3.set_xlim([inds[0]-width*.6, inds[-1]+width*1.4])
  ax3.set_ylabel('MAPE')
  ax3.set_xticklabels(('RecurrentELM23','TM','LSTM','RecurrentELM25', 'RecurrentELM50', 'RecurrentELM100', 'RecurrentELM200', 'RecurrentELM400', 'RecurrentELM800' ))
  for tick in ax3.xaxis.get_major_ticks():
    tick.label.set_rotation('vertical')

  plt.show()
  print (
      altMAPEELM23,
      altMAPETM,
      altMAPELSTM
   #   altMAPEELM100,
   #   altMAPEELM200,
   #   altMAPEELM400,
     # altMAPEELM800,
     # altMAPEELM1600
                 )
  plt.savefig(figPath + 'model_performance_summary_TOTAL.pdf')



  #raw_input()