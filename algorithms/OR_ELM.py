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
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
from FOS_ELM import FOSELM
#from plot import orthogonalization

def orthogonalization(Arr):
  [Q, S, _] = np.linalg.svd(Arr)
  tol = max(Arr.shape) * np.spacing(max(S))
  r = np.sum(S > tol)
  Q = Q[:, :r]


def sigmoidActFunc(features, weights, bias):
  assert(features.shape[1] == weights.shape[1])
  (numSamples, numInputs) = features.shape
  (numHiddenNeuron, numInputs) = weights.shape
  V = np.dot(features, np.transpose(weights)) + bias
  H = 1 / (1+np.exp(-V))
  return H


def linear_recurrent(features, inputW,hiddenW,hiddenA, bias):
  (numSamples, numInputs) = features.shape
  (numHiddenNeuron, numInputs) = inputW.shape
  V = np.dot(features, np.transpose(inputW)) + np.dot(hiddenA,hiddenW) + bias
  return V

def sigmoidAct_forRecurrent(features,inputW,hiddenW,hiddenA,bias):
  (numSamples, numInputs) = features.shape
  (numHiddenNeuron, numInputs) = inputW.shape
  V = np.dot(features, np.transpose(inputW)) + np.dot(hiddenA,hiddenW) + bias
  H = 1 / (1 + np.exp(-V))
  return H

def sigmoidActFunc(V):
  H = 1 / (1+np.exp(-V))
  return H


class ORELM(object):
  def __init__(self, inputs, outputs, numHiddenNeurons, activationFunction, LN=True, AE=True, ORTH=True,
               inputWeightForgettingFactor=0.999,
               outputWeightForgettingFactor=0.999,
               hiddenWeightForgettingFactor=0.999):

    self.activationFunction = activationFunction
    self.inputs = inputs
    self.outputs = outputs
    self.numHiddenNeurons = numHiddenNeurons

    # input to hidden weights
    self.inputWeights  = np.random.random((self.numHiddenNeurons, self.inputs))
    # hidden layer to hidden layer wieghts
    self.hiddenWeights = np.random.random((self.numHiddenNeurons, self.numHiddenNeurons))
    # initial hidden layer activation
    self.initial_H = np.random.random((1, self.numHiddenNeurons)) * 2 -1
    self.H = self.initial_H
    self.LN = LN
    self.AE = AE
    self.ORTH = ORTH
    # bias of hidden units
    self.bias = np.random.random((1, self.numHiddenNeurons)) * 2 - 1
    # hidden to output layer connection
    self.beta = np.random.random((self.numHiddenNeurons, self.outputs))

    # auxiliary matrix used for sequential learning
    self.M = inv(0.00001 * np.eye(self.numHiddenNeurons))

    self.forgettingFactor = outputWeightForgettingFactor

    self.trace=0
    self.thresReset=0.001


    if self.AE:
      self.inputAE = FOSELM(inputs = inputs,
                            outputs = inputs,
                            numHiddenNeurons = numHiddenNeurons,
                            activationFunction = activationFunction,
                            LN= LN,
                            forgettingFactor=inputWeightForgettingFactor,
                            ORTH = ORTH
                            )

      self.hiddenAE = FOSELM(inputs = numHiddenNeurons,
                             outputs = numHiddenNeurons,
                             numHiddenNeurons = numHiddenNeurons,
                             activationFunction=activationFunction,
                             LN= LN,
                             ORTH = ORTH
                             )



  def layerNormalization(self, H, scaleFactor=1, biasFactor=0):

    H_normalized = (H-H.mean())/(np.sqrt(H.var() + 0.000001))
    H_normalized = scaleFactor*H_normalized+biasFactor

    return H_normalized

  def __calculateInputWeightsUsingAE(self, features):
    self.inputAE.train(features=features,targets=features)
    return self.inputAE.beta

  def __calculateHiddenWeightsUsingAE(self, features):
    self.hiddenAE.train(features=features,targets=features)
    return self.hiddenAE.beta

  def calculateHiddenLayerActivation(self, features):
    """
    Calculate activation level of the hidden layer
    :param features feature matrix with dimension (numSamples, numInputs)
    :return: activation level (numSamples, numHiddenNeurons)
    """
    if self.activationFunction is "sig":

      if self.AE:
        self.inputWeights = self.__calculateInputWeightsUsingAE(features)

        self.hiddenWeights = self.__calculateHiddenWeightsUsingAE(self.H)

      V = linear_recurrent(features=features,
                           inputW=self.inputWeights,
                           hiddenW=self.hiddenWeights,
                           hiddenA=self.H,
                           bias= self.bias)
      if self.LN:
        V = self.layerNormalization(V)
      self.H = sigmoidActFunc(V)

    else:
      print " Unknown activation function type"
      raise NotImplementedError
    return self.H


  def initializePhase(self, lamb=0.0001):
    """
    Step 1: Initialization phase
    :param features feature matrix with dimension (numSamples, numInputs)
    :param targets target matrix with dimension (numSamples, numOutputs)
    """



    if self.activationFunction is "sig":
      self.bias = np.random.random((1, self.numHiddenNeurons)) * 2 - 1
    else:
      print " Unknown activation function type"
      raise NotImplementedError

    self.M = inv(lamb*np.eye(self.numHiddenNeurons))
    self.beta = np.zeros([self.numHiddenNeurons,self.outputs])

    # randomly initialize the input->hidden connections
    self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs))
    self.inputWeights = self.inputWeights * 2 - 1

    if self.AE:
     self.inputAE.initializePhase(lamb=0.00001)
     self.hiddenAE.initializePhase(lamb=0.00001)
    else:
      # randomly initialize the input->hidden connections
      self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs))
      self.inputWeights = self.inputWeights * 2 - 1

      if self.ORTH:
        if self.numHiddenNeurons > self.inputs:
          self.inputWeights = orthogonalization(self.inputWeights)
        else:
          self.inputWeights = orthogonalization(self.inputWeights.transpose())
          self.inputWeights = self.inputWeights.transpose()

      # hidden layer to hidden layer wieghts
      self.hiddenWeights = np.random.random((self.numHiddenNeurons, self.numHiddenNeurons))
      self.hiddenWeights = self.hiddenWeights * 2 - 1
      if self.ORTH:
        self.hiddenWeights = orthogonalization(self.hiddenWeights)

  def reset(self):
    self.H = self.initial_H

  def train(self, features, targets,RESETTING=False):
    """
    Step 2: Sequential learning phase
    :param features feature matrix with dimension (numSamples, numInputs)
    :param targets target matrix with dimension (numSamples, numOutputs)
    """
    (numSamples, numOutputs) = targets.shape
    assert features.shape[0] == targets.shape[0]

    H = self.calculateHiddenLayerActivation(features)
    Ht = np.transpose(H)
    try:
      scale = 1/(self.forgettingFactor)
      self.M = scale*self.M - np.dot(scale*self.M,
                       np.dot(Ht, np.dot(
          pinv(np.eye(numSamples) + np.dot(H, np.dot(scale*self.M, Ht))),
          np.dot(H, scale*self.M))))
      if RESETTING:
        beforeTrace=self.trace
        self.trace=self.M.trace()
        print np.abs(beforeTrace - self.trace)
        if np.abs(beforeTrace - self.trace) < self.thresReset:
          print self.M
          eig,_=np.linalg.eig(self.M)
          lambMin=min(eig)
          lambMax=max(eig)
          #lamb = (lambMax+lambMin)/2
          lamb = lambMax
          lamb = lamb.real
          self.M= lamb*np.eye(self.numHiddenNeurons)
          print "reset"
          print self.M

      self.beta = (self.forgettingFactor)*self.beta + np.dot(self.M, np.dot(Ht, targets - np.dot(H, (self.forgettingFactor)*self.beta)))
      #self.beta = self.beta + np.dot(self.M, np.dot(Ht, targets - np.dot(H, self.beta)))


    except np.linalg.linalg.LinAlgError:
      print "SVD not converge, ignore the current training cycle"
    # else:
    #   raise RuntimeError

  def predict(self, features):
    """
    Make prediction with feature matrix
    :param features: feature matrix with dimension (numSamples, numInputs)
    :return: predictions with dimension (numSamples, numOutputs)
    """
    H = self.calculateHiddenLayerActivation(features)
    prediction = np.dot(H, self.beta)
    return prediction

