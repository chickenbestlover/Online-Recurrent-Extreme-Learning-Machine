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
# And modified to run Online Recurrent Extreme Learning Machine (OR-ELM)
# ----------------------------------------------------------------------
from matplotlib import pyplot as plt
from errorMetrics import *
import pandas as pd
import datetime
from pylab import rcParams
from plot import plotAccuracy, computeSquareDeviation
import os.path

rcParams.update({'figure.autolayout': True})
rcParams.update({'figure.facecolor': 'white'})
rcParams.update({'ytick.labelsize': 8})
rcParams.update({'figure.figsize': (12, 6)})
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
plt.ion()
plt.close('all')

window = 960
skipTrain = 10000
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


def loadExperimentResult(filePath,predictionStep=1):
  expResult = pd.read_csv(filePath, header=0, skiprows=[1, 2],
                            names=['step', 'value', 'prediction'+str(predictionStep)])
  groundTruth = np.roll(expResult['value'], -predictionStep)
  predictions = np.array(expResult['prediction'+str(predictionStep)])
  return (groundTruth, predictions)

def load_plot_return_errors(forgettingFactor, algorithmname, nHidden, dataset='nyc_taxi', skipTrain=500, window=100):
    forgettingFactor = str(forgettingFactor)
    nHidden = str(nHidden)
    filepath = './prediction/' + dataset + '_FF' + forgettingFactor + algorithmname + nHidden + '_pred.csv'
    if os.path.isfile(filepath):
      (elmTruth, elmPrediction) = loadExperimentResult(filepath)
      squareDeviation = computeSquareDeviation(elmPrediction, elmTruth)
      squareDeviation[:skipTrain] = None
      nrse_temp = plotAccuracy((squareDeviation, xaxisDate),
                                  elmTruth,
                                  window=window,
                                  errorType='square_deviation',
                                  label=algorithmname+'_#HN='+nHidden)
      nrse_temp = np.sqrt(np.nanmean(nrse_temp)) / np.nanstd(elmTruth)
      nrse_temp = min(1000,nrse_temp)
      mape_temp = computeAltMAPE(elmTruth, elmPrediction, skipTrain)
      mape_temp = min(1000,mape_temp)
    else:
      nrse_temp=0
      mape_temp=0
    return [nrse_temp,mape_temp]

if __name__ == "__main__":
  xaxisDate = getDatetimeAxis()
  plt.figure(figsize=(8,4))
  dataSet = 'nyc_taxi'

  forgettingFactor = 0.915

  nrseFOSELM=[]
  mapeFOSELM=[]

  algorithmname1= 'FOSELM'
  print 'forgettingFactor = '+str(forgettingFactor) + '  algorithm = ' + algorithmname1
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname1, nHidden=10, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseFOSELM.append(nrse_temp)
  mapeFOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname1, nHidden=20, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseFOSELM.append(nrse_temp)
  mapeFOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname1, nHidden=25, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseFOSELM.append(nrse_temp)
  mapeFOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname1, nHidden=50, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseFOSELM.append(nrse_temp)
  mapeFOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname1, nHidden=100, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseFOSELM.append(nrse_temp)
  mapeFOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname1, nHidden=200, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseFOSELM.append(nrse_temp)
  mapeFOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname1, nHidden=400, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseFOSELM.append(nrse_temp)
  mapeFOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname1, nHidden=800, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseFOSELM.append(nrse_temp)
  mapeFOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname1, nHidden=1600, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseFOSELM.append(nrse_temp)
  mapeFOSELM.append(mape_temp)

  nrseNFOSELM=[]
  mapeNFOSELM=[]
  algorithmname2= 'NFOSELM'
  print 'forgettingFactor = '+str(forgettingFactor) + '  algorithm = ' + algorithmname1
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname2, nHidden=10, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseNFOSELM.append(nrse_temp)
  mapeNFOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname2, nHidden=20, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseNFOSELM.append(nrse_temp)
  mapeNFOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname2, nHidden=25, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseNFOSELM.append(nrse_temp)
  mapeNFOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname2, nHidden=50, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseNFOSELM.append(nrse_temp)
  mapeNFOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname2, nHidden=100, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseNFOSELM.append(nrse_temp)
  mapeNFOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname2, nHidden=200, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseNFOSELM.append(nrse_temp)
  mapeNFOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname2, nHidden=400, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseNFOSELM.append(nrse_temp)
  mapeNFOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname2, nHidden=800, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseNFOSELM.append(nrse_temp)
  mapeNFOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname2, nHidden=1600, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseNFOSELM.append(nrse_temp)
  mapeNFOSELM.append(mape_temp)

  nrseNAOSELM=[]
  mapeNAOSELM=[]
  algorithmname3= 'NAOSELM'
  print 'forgettingFactor = '+str(forgettingFactor) + '  algorithm = ' + algorithmname1
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname3, nHidden=10, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseNAOSELM.append(nrse_temp)
  mapeNAOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname3, nHidden=20, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseNAOSELM.append(nrse_temp)
  mapeNAOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname3, nHidden=25, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseNAOSELM.append(nrse_temp)
  mapeNAOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname3, nHidden=50, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseNAOSELM.append(nrse_temp)
  mapeNAOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname3, nHidden=100, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseNAOSELM.append(nrse_temp)
  mapeNAOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname3, nHidden=200, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseNAOSELM.append(nrse_temp)
  mapeNAOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname3, nHidden=400, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseNAOSELM.append(nrse_temp)
  mapeNAOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname3, nHidden=800, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseNAOSELM.append(nrse_temp)
  mapeNAOSELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname3, nHidden=1600, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseNAOSELM.append(nrse_temp)
  mapeNAOSELM.append(mape_temp)

  nrseORELM=[]
  mapeORELM=[]
  algorithmname4= 'ORELM'
  print 'forgettingFactor = '+str(forgettingFactor) + '  algorithm = ' + algorithmname1
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname4, nHidden=10, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseORELM.append(nrse_temp)
  mapeORELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname4, nHidden=20, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseORELM.append(nrse_temp)
  mapeORELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname4, nHidden=25, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseORELM.append(nrse_temp)
  mapeORELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname4, nHidden=50, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseORELM.append(nrse_temp)
  mapeORELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname4, nHidden=100, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseORELM.append(nrse_temp)
  mapeORELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname4, nHidden=200, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseORELM.append(nrse_temp)
  mapeORELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname4, nHidden=400, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseORELM.append(nrse_temp)
  mapeORELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname4, nHidden=800, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseORELM.append(nrse_temp)
  mapeORELM.append(mape_temp)
  [nrse_temp, mape_temp]=load_plot_return_errors(forgettingFactor=forgettingFactor, algorithmname=algorithmname4, nHidden=1600, dataset='nyc_taxi', skipTrain=skipTrain)
  nrseORELM.append(nrse_temp)
  mapeORELM.append(mape_temp)

  #plt.xlim([datetime.datetime(2014,7,13,12,0),datetime.datetime(2014,7,15,0,0)])
  #plt.xlim([datetime.datetime(2014, 7, 13, 12, 0), datetime.datetime(2014, 9, 12, 0, 0)])
  plt.xlim([datetime.datetime(2015, 3, 30, 12, 0), datetime.datetime(2015, 5, 1, 0, 0)])

  plt.ylim([0.0,1.5])
  #plt.ylim([0.2,0.9])
  #plt.legend(loc='lower right')
  plt.legend(loc='lower right')
  plt.legend()
  plt.savefig(figPath + 'squrare_deviation_FF' + str(forgettingFactor) + 'FOSELMvsNFOSELMvsNAOSELMvsORELM.pdf')

  print nrseORELM

  fig, ax = plt.subplots(nrows=1, ncols=2)
  inds = np.arange(len(nrseFOSELM))
  ax1 = ax[0]
  width = 0.15
  rects1 = ax1.bar(inds, nrseFOSELM, width=width, color='r')
  rects2 = ax1.bar(inds + width, nrseNFOSELM, width=width, color ='g')
  rects3 = ax1.bar(inds + 2 * width, nrseNAOSELM, width=width, color ='b')
  rects4 = ax1.bar(inds + 3 * width, nrseORELM, width=width, color ='gray')
  ax1.set_xticks(inds+width)
  ax1.set_ylabel('NRMSE')
  ax1.set_xlim([inds[0]-width, inds[-1]+width*5.0])
  ax1.set_xticklabels( ('10','20','25','50','100','200','400','800','1600') )
  ax1.set_xlabel('# of hidden neurons')
  ax1.legend((rects1[0], rects2[0], rects3[0],rects4[0]), ('FOSELM', 'NFOSELM', 'NAOSELM','ORELM'))
  ax1.set_ylim([0,0.8])
  '''
  for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_rotation('vertical')
  '''
  ax3 = ax[1]
  rects11 = ax3.bar(inds, mapeFOSELM, width=width, color='r')
  rects12 = ax3.bar(inds + width, mapeNFOSELM, width=width, color ='g')
  rects13 = ax3.bar(inds + 2 * width, mapeNAOSELM, width=width, color ='b')
  rects14 = ax3.bar(inds + 3 * width, mapeORELM, width=width, color='gray')

  ax3.set_xticks(inds+width)
  ax3.set_xlim([inds[0]-width, inds[-1]+width*5.0])
  ax3.set_ylabel('MAPE')
  ax3.set_xticklabels(('10','20','25','50','100','200','400','800','1600') )
  ax3.set_xlabel('# of hidden neurons')
  ax3.legend((rects11[0], rects12[0], rects13[0],rects14[0]), ('FOSELM', 'NFOSELM', 'NAOSELM','ORELM'))
  ax3.set_ylim([0, 0.3])
  '''
  for tick in ax3.xaxis.get_major_ticks():
    tick.label.set_rotation('vertical')
  '''
  plt.show()
  print 'mapeFOSELM ='
  print (mapeFOSELM)
  print 'mapeNFOSELM ='
  print (mapeNFOSELM)
  print 'mapeNAOSELM ='
  print (mapeNAOSELM)
  print 'mapeORELM ='
  print (mapeORELM)

  plt.savefig(figPath + 'model_performance_summary_FF' + str(forgettingFactor) + 'FOSELMvsNFOSELMvsNAOSELMvsORELM.pdf')


  #plt.figure()


  #raw_input()