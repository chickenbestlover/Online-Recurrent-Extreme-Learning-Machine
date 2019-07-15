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

def linear(features, weights, bias):
  assert(features.shape[1] == weights.shape[1])
  (numSamples, numInputs) = features.shape
  (numHiddenNeuron, numInputs) = weights.shape
  V = np.dot(features, np.transpose(weights)) + bias

  return V

def sigmoidActFunc(V):
  H = 1 / (1+np.exp(-V))
  return H

class NAOSELM(object):
  def __init__(self, inputs, outputs, numHiddenNeurons, activationFunction, LN=True,
               outputWeightForgettingFactor=0.999,
               inputWeightForgettingFactor=0.999, AE=True, ORTH=False):


    self.activationFunction = activationFunction
    self.inputs = inputs
    self.outputs = outputs
    self.numHiddenNeurons = numHiddenNeurons

    # input to hidden weights
    self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs))
    # bias of hidden units
    #self.bias = np.random.random((1, self.numHiddenNeurons)) * 2 - 1

    self.bias = np.zeros([1,self.numHiddenNeurons])
    # hidden to output layer connection
    self.beta = np.random.random((self.numHiddenNeurons, self.outputs))

    # auxiliary matrix used for sequential learning
    self.M = None
    self.LN = LN
    self.AE = AE
    self.ORTH = ORTH
    self.forgettingFactor =outputWeightForgettingFactor
    self.FOSELM_AE = FOSELM(inputs= inputs,
                            outputs = inputs,
                            numHiddenNeurons= numHiddenNeurons,
                            activationFunction= activationFunction,
                            LN=True,
                            forgettingFactor=inputWeightForgettingFactor,
                            ORTH=self.ORTH)


  def layerNormalization(self, H, scaleFactor=1, biasFactor=0):

    H_normalized = (H-H.mean())/(np.sqrt(H.var() + 0.00001))
    H_normalized = scaleFactor*H_normalized+biasFactor

    return H_normalized

  def __calculateInputWeightsUsingAE(self, features):
    self.FOSELM_AE.train(features=features,targets=features)
    return self.FOSELM_AE.beta

  def calculateHiddenLayerActivation(self, features):
    """
    Calculate activation level of the hidden layer
    :param features feature matrix with dimension (numSamples, numInputs)
    :return: activation level (numSamples, numHiddenNeurons)
    """
    if self.AE:
      self.inputWeights = self.__calculateInputWeightsUsingAE(features)
    if self.activationFunction is "sig":
      V = linear(features, self.inputWeights,self.bias)
      if self.LN:
        V = self.layerNormalization(V)
      H = sigmoidActFunc(V)
    else:
      print " Unknown activation function type"
      raise NotImplementedError

    return H

  def initializePhase(self, lamb=0.0001):
    """
    Step 1: Initialization phase
    :param features feature matrix with dimension (numSamples, numInputs)
    :param targets target matrix with dimension (numSamples, numOutputs)
    """

    self.FOSELM_AE.initializePhase(lamb=lamb)
    self.M = inv(lamb*np.eye(self.numHiddenNeurons))
    self.beta = np.zeros([self.numHiddenNeurons,self.outputs])


  def train(self, features, targets,VFF_RLS=False):
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

      #self.beta = (self.forgettingFactor)*self.beta + np.dot(self.M, np.dot(Ht, targets - np.dot(H, (self.forgettingFactor)*self.beta)))
      self.beta = self.beta + np.dot(self.M, np.dot(Ht, targets - np.dot(H, self.beta)))

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

