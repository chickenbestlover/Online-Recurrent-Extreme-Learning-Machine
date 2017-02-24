import csv
from optparse import OptionParser
from matplotlib import pyplot as plt
import pandas as pd
from errorMetrics import *

from algorithms.FOS_ELM import FOSELM



def readDataSet(dataSet):
  filePath = 'data/'+dataSet+'.csv'

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
  else:
    raise(' unrecognized dataset type ')

  return seq


def getTimeEmbeddedMatrix(sequence, numLags=100, predictionStep=1,
                      useTimeOfDay=True, useDayOfWeek=True):
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
                    help="DataSet Name, choose from nyc_taxi")

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
    row = csvReader.next()
    csvWriter.writerow([row[0], row[1], predictedInput[i]])

  inputFile.close()
  outputFile.close()



if __name__ == "__main__":

  (_options, _args) = _getArgs()
  dataSet = _options.dataSet
  predictionStep = _options.predictionStep

  print "run ELM on ", dataSet

  #predictionStep = 5
  useTimeOfDay = False
  useDayOfWeek = False
  nTrain = 500
  numLags = 100

  # prepare dataset
  sequence = readDataSet(dataSet)

  # standardize data by subtracting mean and dividing by std
  meanSeq = np.mean(sequence['data'])
  stdSeq = np.std(sequence['data'])
  sequence['data'] = (sequence['data'] - meanSeq)/stdSeq

  meanTimeOfDay = np.mean(sequence['timeofday'])
  stdTimeOfDay = np.std(sequence['timeofday'])
  sequence['timeofday'] = (sequence['timeofday'] - meanTimeOfDay)/stdTimeOfDay

  meanDayOfWeek = np.mean(sequence['dayofweek'])
  stdDayOfWeek = np.std(sequence['dayofweek'])
  sequence['dayofweek'] = (sequence['dayofweek'] - meanDayOfWeek)/stdDayOfWeek

  (X, T) = getTimeEmbeddedMatrix(sequence, numLags, predictionStep,
                                 useTimeOfDay, useDayOfWeek)

  #random.seed(7)

  net = FOSELM(X.shape[1], 1,
            numHiddenNeurons=23, activationFunction='sig',BN=True,forgettingFactor=0.915,VFF_RLS=False,ADAPT=False)
  net.initializePhase(lamb = 0.0001)



  predictedInput = np.zeros((len(sequence),))
  targetInput = np.zeros((len(sequence),))
  trueData = np.zeros((len(sequence),))



  for i in xrange(nTrain, len(sequence)-predictionStep-1):
    net.train(X[[i], :], T[[i], :],RLS=False,RESETTING=False)
    Y = net.predict(X[[i+1], :])

    predictedInput[i+1] = Y[-1]
    targetInput[i+1] = sequence['data'][i+1+predictionStep]
    trueData[i+1] = sequence['data'][i+1]
    print "Iteration {} target input {:2.2f} predicted Input {:2.2f} ".format(
      i, targetInput[i+1], predictedInput[i+1])

  predictedInput = (predictedInput * stdSeq) + meanSeq
  targetInput = (targetInput * stdSeq) + meanSeq
  trueData = (trueData * stdSeq) + meanSeq

  saveResultToFile(dataSet, predictedInput, 'Decay'+str(net.forgettingFactor)+'OSELM'+str(net.numHiddenNeurons))

  plt.figure()
  targetPlot,=plt.plot(targetInput,label='target',color='red')
  predictedPlot,=plt.plot(predictedInput,label='predicted',color='blue')
  plt.xlim([0,13500])
  plt.ylim([0, 30000])
  plt.ylabel('')
  plt.xlabel('')
  plt.legend(handles=[targetPlot, predictedPlot])

  #plt.draw()
  #plt.show()


  skipTrain = 500
  from plot import computeSquareDeviation
  squareDeviation = computeSquareDeviation(predictedInput, targetInput)
  squareDeviation[:skipTrain] = None
  nrmse = np.sqrt(np.nanmean(squareDeviation)) / np.nanstd(targetInput)
  print "NRMSE {}".format(nrmse)


  #raw_input()
