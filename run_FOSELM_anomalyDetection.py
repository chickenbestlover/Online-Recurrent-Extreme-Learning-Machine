import csv
from optparse import OptionParser
from matplotlib import pyplot as plt
import datetime
import scipy.special
import pandas as pd
from errorMetrics import *
from algorithms.FOS_ELM import FOSELM
import random
#plt.ion()

qfunc = lambda x: 0.5-0.5*scipy.special.erf(x/np.sqrt(2))

def readDataSet(dataSet,numAnomaly=1):
  filePath = 'data/'+dataSet+'.csv'
  dt_mean = 0
  dt_std = 0

  if dataSet=='nyc_taxi':
    df = pd.read_csv(filePath, header=0, skiprows=[1,2],
                     names=['time', 'data', 'timeofday', 'dayofweek'])
    sequence = df['data']
    dayofweek = df['dayofweek']
    timeofday = df['timeofday']

    seq = pd.DataFrame(np.array(pd.concat([sequence, timeofday, dayofweek], axis=1)),
                        columns=['data', 'timeofday', 'dayofweek'])
  elif dataSet=='sine':
    df = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['time', 'data'])
    sequence = df['data']
    seq = pd.DataFrame(np.array(sequence), columns=['data'])
  elif dataSet == 'kohyoung':
    df = pd.read_csv(filePath, header=None)
    dt = []
    prevTime = datetime.datetime.today()
    for i in xrange(len(df)):
      df_splited = df.loc[i, 0].split()
      df_splited[0] = df_splited[0][1:]
      df_date = [df_splited[0][:4], df_splited[0][4:6], df_splited[0][6:]]
      df_splited[1] = df_splited[1][:len(df_splited[1]) - 2]
      df_time = df_splited[1].split(':')
      temp_dt = datetime.datetime(int(df_date[0]), int(df_date[1]), int(df_date[2]),
                                  int(df_time[0]), int(df_time[1]), int(df_time[2]), int(df_time[3]))
      if i > 0:
        time_diff = (temp_dt - prevTime).total_seconds()
        dt.append(time_diff)
      prevTime = temp_dt
    dt_mean = np.mean(dt)
    dt_std = np.std(dt)
    dt = (dt - dt_mean) / dt_std
    for i in xrange(5):
      dt = np.concatenate((dt, dt), axis=0)
    noise = np.random.randn(len(dt))
    dt = dt + noise / 100

    df1 = pd.read_csv('data/anomaly/anomaly_'+str(numAnomaly)+'.mcslog',header=None)
    dt1 = []
    prevTime = datetime.datetime.today()
    for i in xrange(len(df1)):
      df_splited = df1.loc[i, 0].split()
      df_splited[0] = df_splited[0][1:]
      df_date = [df_splited[0][:4], df_splited[0][4:6], df_splited[0][6:]]
      df_splited[1] = df_splited[1][:len(df_splited[1]) - 2]
      df_time = df_splited[1].split(':')
      temp_dt = datetime.datetime(int(df_date[0]), int(df_date[1]), int(df_date[2]),
                                  int(df_time[0]), int(df_time[1]), int(df_time[2]), int(df_time[3]))
      if i > 0:
        time_diff = (temp_dt - prevTime).total_seconds()
        dt1.append(time_diff)
      prevTime = temp_dt
    dt1 = (dt1 - dt_mean) / dt_std

    dt = np.concatenate((dt,dt1),axis=0)
    seq = pd.DataFrame(dt, columns=['data'])
  else:
    raise(' unrecognized dataset type ')

  return seq, dt_mean, dt_std


def getTimeEmbeddedMatrix(sequence, numLags=100, predictionStep=1,
                      useTimeOfDay=False, useDayOfWeek=False):
  print "generate time embedded matrix "
  print "the training data contains ", str(nTrain-predictionStep), "records"

  inDim = numLags + int(useTimeOfDay) + int(useDayOfWeek)

  if useTimeOfDay:
    print "include time of day as input field"
  if useDayOfWeek:
    print "include day of week as input field"

  X = np.zeros(shape=(len(sequence), inDim))
  T = np.zeros(shape=(len(sequence), 1))
  for i in xrange(numLags-1, len(sequence)-predictionStep):
    if useTimeOfDay and useDayOfWeek:
      sample = np.concatenate([np.array(sequence['data'][(i-numLags+1):(i+1)]),
                               np.array([sequence['timeofday'][i]]),
                               np.array([sequence['dayofweek'][i]])])
    elif useTimeOfDay:
      sample = np.concatenate([np.array(sequence['data'][(i-numLags+1):(i+1)]),
                               np.array([sequence['timeofday'][i]])])
    elif useDayOfWeek:
      sample = np.concatenate([np.array(sequence['data'][(i-numLags+1):(i+1)]),
                               np.array([sequence['dayofweek'][i]])])
    else:
      sample = np.array(sequence['data'][(i-numLags+1):(i+1)])

    X[i, :] = sample
    T[i, :] = sequence['data'][i+predictionStep]

  return (X, T)



def _getArgs():
  parser = OptionParser(usage="%prog PARAMS_DIR OUTPUT_DIR [options]"
                              "\n\nCompare TM performance with trivial predictor using "
                              "model outputs in prediction directory "
                              "and outputting results to result directory.")
  parser.add_option("-d",
                    "--dataSet",
                    type=str,
                    default='nyc_taxi',
                    dest="dataSet",
                    help="DataSet Name, choose among nyc_taxi, kohyoung")

  parser.add_option("-n",
                     "--predictionStep",
                     type=int,
                     default=5,
                     dest="predictionStep",
                     help="number of steps ahead to be predicted")


  (options, remainder) = parser.parse_args()
  print options

  return options, remainder



def saveResultToFile(dataSet, predictedInput, algorithmName):
  inputFileName = 'data/' + dataSet + '.csv'
  inputFile = open(inputFileName, "rb")

  csvReader = csv.reader(inputFile)

  # skip header rows
  csvReader.next()
  csvReader.next()
  csvReader.next()

  outputFileName = './prediction/' + dataSet + '_' + algorithmName + '_pred.csv'
  outputFile = open(outputFileName, "w")
  csvWriter = csv.writer(outputFile)
  csvWriter.writerow(
    ['timestamp', 'data', 'prediction-' + str(predictionStep) + 'step'])
  csvWriter.writerow(['datetime', 'float', 'float'])
  csvWriter.writerow(['', '', ''])

  for i in xrange(len(sequence)):

    if dataSet == 'kohyoung':
      pass
    else:
      row = csvReader.next()
      csvWriter.writerow([row[0], row[1], predictedInput[i]])

  inputFile.close()
  outputFile.close()



if __name__ == "__main__":

  (_options, _args) = _getArgs()
  dataSet = _options.dataSet
  predictionStep = _options.predictionStep

  print "run ELM on ", dataSet
  useTimeOfDay = False
  useDayOfWeek = False

  nTrain = 500
  numLags = 100

  # prepare dataset as pyBrain sequential dataset
  numAnomaly=11
  sequence, __, __ = readDataSet(dataSet,numAnomaly=numAnomaly)

  # standardize data by subtracting mean and dividing by std
  if dataSet=='kohyoung':
    sequence, meanSeq, stdSeq = readDataSet(dataSet, numAnomaly=numAnomaly)

  else:

    meanSeq = np.mean(sequence['data'])
    stdSeq = np.std(sequence['data'])
    sequence['data'] = (sequence['data'] - meanSeq)/stdSeq

  if useTimeOfDay:
    meanTimeOfDay = np.mean(sequence['timeofday'])
    stdTimeOfDay = np.std(sequence['timeofday'])
    sequence['timeofday'] = (sequence['timeofday'] - meanTimeOfDay)/stdTimeOfDay
  if useDayOfWeek:
    meanDayOfWeek = np.mean(sequence['dayofweek'])
    stdDayOfWeek = np.std(sequence['dayofweek'])
    sequence['dayofweek'] = (sequence['dayofweek'] - meanDayOfWeek)/stdDayOfWeek

  (X, T) = getTimeEmbeddedMatrix(sequence, numLags, predictionStep,
                                 useTimeOfDay, useDayOfWeek)

  random.seed(6)
  net = FOSELM(X.shape[1], 1,
            numHiddenNeurons=23, activationFunction='sig',BN=True,forgettingFactor=0.915,ORTH=False)
  net.initializePhase(lamb = 0.00001)

  predictedInput = np.zeros((len(sequence),))
  targetInput = np.zeros((len(sequence),))
  trueData = np.zeros((len(sequence),))
  anomalyScore = np.zeros((len(sequence),))
  anomalyLikelihood = np.zeros((len(sequence),))
  a = np.exp(anomalyLikelihood)
  Ascores =[]
  ASpoints = []
  ALpoints = []

  for i in xrange(nTrain, len(sequence)-predictionStep-1):
    net.train(X[[i], :], T[[i], :],RLS=True)
    Y = net.predict(X[[i+1], :])

    predictedInput[i+1] = Y[-1]
    targetInput[i+1] = sequence['data'][i+1+predictionStep]
    trueData[i+1] = sequence['data'][i+1]
    #print "Iteration {} target input {:2.2f} predicted Input {:2.2f} ".format(i, targetInput[i+1], predictedInput[i+1])
    anomalyScore[i+1] = np.abs(targetInput[i+1]-predictedInput[i+1])/np.abs(targetInput[i+1])
    Ascores.append(anomalyScore[i+1])
    #print i,"th prediction : anomalyScore = ",anomalyScore
    if anomalyScore[i+1]>0.7:
      #print "Anomaly!!!!!!!!!!"
      ASpoints.append(i)
    meanLongA = np.mean(anomalyScore[i+1-500:i+1])
    stdLongA = np.std(anomalyScore[i+1-500:i+1])
    meanShortA = np.mean(anomalyScore[i+1-10:i+1])
    anomalyLikelihood[i+1]= 1-qfunc((meanShortA-meanLongA)/(stdLongA+0.00001))
    if (anomalyLikelihood[i+1]>0.9):
      print i, "th prediction : AL = ", round(anomalyLikelihood[i+1],4), "  \t AS= ", round(anomalyScore[i+1],4)\
      , "  \t Target= ", round(targetInput[i+1],4), "  \t Output= ", round(predictedInput[i+1],4)


    if anomalyLikelihood[i+1]>1-0.001:
      ALpoints.append(i)

  predictedInput = (predictedInput * stdSeq) + meanSeq
  targetInput = (targetInput * stdSeq) + meanSeq
  trueData = (trueData * stdSeq) + meanSeq

  saveResultToFile(dataSet, predictedInput, 'FOSELM'+str(net.numHiddenNeurons))

  plt.close('all')

  f, axarr = plt.subplots(2, sharex=True)

  targetPlot,=axarr[0].plot(targetInput,label='target',color='red',alpha=0.5)
  predictedPlot,=axarr[0].plot(predictedInput,label='predicted',color='blue',alpha=0.5)
  anomalyPoints,=axarr[0].plot([],label='anomalies',color='green')
  for i in ALpoints:
    sc = axarr[0].axvspan(i, i + 1, facecolor='yellow', edgecolor='yellow', label='anomalies')

  axarr[0].set_ylim([0,30])
  axarr[0].set_ylabel('')
  axarr[0].set_xlabel('')
  axarr[0].legend(handles=[targetPlot, predictedPlot,sc])

  for i in ALpoints:
    sc = axarr[1].axvspan(i, i + 1, facecolor='yellow', alpha=0.5, edgecolor='yellow', label='anomalies')


  #sc = axarr[1].scatter(ALpoints, np.zeros(len(ALpoints)), label='anomalies', marker='o', color='black', s=200)
  asPlot, = axarr[1].plot((anomalyLikelihood), label='ALikelihood', color='red', alpha=0.5)
  axarr[1].set_ylim([0.9, 1.1])

  axarr[1].legend(handles=[asPlot,sc])

  plt.savefig('result/anomaly_'+str(numAnomaly)+'_plot.pdf')
  plt.show()
  skipTrain = 1000
  skipTrain2 = 10000
  from plot import computeSquareDeviation
  squareDeviation = computeSquareDeviation(predictedInput, targetInput)
  squareDeviation[:skipTrain] = None
  squareDeviation[skipTrain2:] = None
  nrmse = np.sqrt(np.nanmean(squareDeviation)) / np.nanstd(targetInput)
  print "NRMSE {}".format(nrmse)


  #raw_input()
