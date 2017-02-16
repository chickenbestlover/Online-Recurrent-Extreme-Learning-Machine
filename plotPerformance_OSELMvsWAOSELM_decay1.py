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

window = 10
skipTrain = 6000
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

def load_plot_return_errors(decay,algorithmname,nHidden,dataset='nyc_taxi',skipTrain=500,window=10):
    decay = str(decay)
    nHidden = str(nHidden)
    (elmTruth, elmPrediction) = loadExperimentResult('./prediction/' + dataSet + '_Decay'+decay+algorithmname+nHidden+'_pred.csv')
    squareDeviation = computeSquareDeviation(elmPrediction, elmTruth)
    squareDeviation[:skipTrain] = None
    nrseOSELM_temp = plotAccuracy((squareDeviation, xaxisDate),
                                elmTruth,
                                window=window,
                                errorType='square_deviation',
                                label=algorithmname+'_#HN='+nHidden)
    nrseOSELM_temp = np.sqrt(np.nanmean(nrseOSELM_temp)) / np.nanstd(elmTruth)
    mapeOSELM_temp = computeAltMAPE(elmTruth, elmPrediction, skipTrain)
    return [nrseOSELM_temp,mapeOSELM_temp]

if __name__ == "__main__":
  xaxisDate = getDatetimeAxis()
  plt.figure(figsize=(8,4))
  dataSet = 'nyc_taxi'

  nrseOSELM=[]
  mapeOSELM=[]
  decay=1
  algorithmname='OSELM'
  print 'decay = '+str(decay)+'  algorithm = '+algorithmname
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=10,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseOSELM.append(nrseOSELM_temp)
  mapeOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=20,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseOSELM.append(nrseOSELM_temp)
  mapeOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=25,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseOSELM.append(nrseOSELM_temp)
  mapeOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=50,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseOSELM.append(nrseOSELM_temp)
  mapeOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=100,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseOSELM.append(nrseOSELM_temp)
  mapeOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=200,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseOSELM.append(nrseOSELM_temp)
  mapeOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=400,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseOSELM.append(nrseOSELM_temp)
  mapeOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=800,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseOSELM.append(nrseOSELM_temp)
  mapeOSELM.append(mapeOSELM_temp)

  nrseWOSELM=[]
  mapeWOSELM=[]
  decay=1
  algorithmname='WOSELM'
  print 'decay = '+str(decay)+'  algorithm = '+algorithmname
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=10,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseWOSELM.append(nrseOSELM_temp)
  mapeWOSELM.append(mapeOSELM_temp)
  [nrseWOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=20,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseWOSELM.append(nrseOSELM_temp)
  mapeWOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=25,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseWOSELM.append(nrseOSELM_temp)
  mapeWOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=50,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseWOSELM.append(nrseOSELM_temp)
  mapeWOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=100,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseWOSELM.append(nrseOSELM_temp)
  mapeWOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=200,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseWOSELM.append(nrseOSELM_temp)
  mapeWOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=400,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseWOSELM.append(nrseOSELM_temp)
  mapeWOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=800,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseWOSELM.append(nrseOSELM_temp)
  mapeWOSELM.append(mapeOSELM_temp)

  nrseWAOSELM=[]
  mapeWAOSELM=[]
  decay=1
  algorithmname='WAOSELM'
  print 'decay = '+str(decay)+'  algorithm = '+algorithmname
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=10,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseWAOSELM.append(nrseOSELM_temp)
  mapeWAOSELM.append(mapeOSELM_temp)
  [nrseWAOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=20,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseWAOSELM.append(nrseOSELM_temp)
  mapeWAOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=25,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseWAOSELM.append(nrseOSELM_temp)
  mapeWAOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=50,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseWAOSELM.append(nrseOSELM_temp)
  mapeWAOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=100,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseWAOSELM.append(nrseOSELM_temp)
  mapeWAOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=200,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseWAOSELM.append(nrseOSELM_temp)
  mapeWAOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=400,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseWAOSELM.append(nrseOSELM_temp)
  mapeWAOSELM.append(mapeOSELM_temp)
  [nrseOSELM_temp,mapeOSELM_temp]=load_plot_return_errors(decay=decay,algorithmname=algorithmname,nHidden=800,dataset='nyc_taxi',skipTrain=skipTrain)
  nrseWAOSELM.append(nrseOSELM_temp)
  mapeWAOSELM.append(mapeOSELM_temp)


  #plt.xlim([datetime.datetime(2014,7,13,12,0),datetime.datetime(2014,7,15,0,0)])
  #plt.xlim([datetime.datetime(2014, 7, 13, 12, 0), datetime.datetime(2014, 9, 12, 0, 0)])

  plt.ylim([0.0,1.5])
  #plt.ylim([0.2,0.9])
  #plt.legend(loc='lower right')
  plt.legend(loc='lower right')
  plt.legend()
  plt.savefig(figPath + 'squrare_deviation_Decay'+str(decay)+'OSELMvsWAOSELM.pdf')

  startFrom = skipTrain


  fig, ax = plt.subplots(nrows=1, ncols=2)
  inds = np.arange(len(nrseOSELM))
  ax1 = ax[0]
  width = 0.20
  rects1 = ax1.bar(inds, nrseOSELM, width=width, color='r')
  rects2 = ax1.bar(inds+width, nrseWOSELM, width=width, color = 'g')
  rects3 = ax1.bar(inds+2*width, nrseWAOSELM, width=width, color = 'b')
  ax1.set_xticks(inds+width)
  ax1.set_ylabel('NRMSE')
  ax1.set_xlim([inds[0]-width, inds[-1]+width*4.0])
  ax1.set_xticklabels( ('10','20','25','50','100','200','400','800','1600') )
  ax1.set_xlabel('# of hidden neurons')
  ax1.legend((rects1[0], rects2[0], rects3[0]), ('OSELM', 'WOSELM', 'WAOSELM'))

  '''
  for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_rotation('vertical')
  '''
  ax3 = ax[1]
  rects3 = ax3.bar(inds, mapeOSELM, width=width, color='r')
  rects4 = ax3.bar(inds+width, mapeWOSELM, width=width, color = 'g')
  rects5 = ax3.bar(inds+2*width, mapeWAOSELM, width=width, color = 'b')
  ax3.set_xticks(inds+width)
  ax3.set_xlim([inds[0]-width, inds[-1]+width*4.0])
  ax3.set_ylabel('MAPE')
  ax3.set_xticklabels(('10','20','25','50','100','200','400','800','1600') )
  ax3.set_xlabel('# of hidden neurons')
  ax3.legend((rects3[0], rects4[0], rects5[0]), ('OSELM', 'WOSELM', 'WAOSELM'))

  '''
  for tick in ax3.xaxis.get_major_ticks():
    tick.label.set_rotation('vertical')
  '''
  plt.show()
  print 'mapeOSELM ='
  print (mapeOSELM)
  print 'mapeWOSELM ='
  print (mapeWOSELM)
  print 'mapeWAOSELM ='
  print (mapeWAOSELM)

  plt.savefig(figPath + 'model_performance_summary_Decay'+str(decay)+'OSELMvsWOSELMvsWAOSELM.pdf')



  #raw_input()