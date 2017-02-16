# coding=utf-8
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

import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
"""
Implementation of the online-sequential extreme learning machine

Reference:
N.-Y. Liang, G.-B. Huang, P. Saratchandran, and N. Sundararajan,
“A Fast and Accurate On-line Sequential Learning Algorithm for Feedforward
Networks," IEEE Transactions on Neural Networks, vol. 17, no. 6, pp. 1411-1423
"""


def linear(features, weights, bias):
  assert(features.shape[1] == weights.shape[1])
  (numSamples, numInputs) = features.shape
  (numHiddenNeuron, numInputs) = weights.shape
  V = np.dot(features, np.transpose(weights))
  for i in range(numHiddenNeuron):
    V[:, i] += bias[0, i]

  return V

def sigmoidActFunc(V):
  H = 1 / (1+np.exp(-V))
  return H

class OSELM(object):
  def __init__(self, inputs, outputs, numHiddenNeurons, activationFunction,BN=True,forgettingFactor=0.99):

    self.activationFunction = activationFunction
    self.inputs = inputs
    self.outputs = outputs
    self.numHiddenNeurons = numHiddenNeurons

    # input to hidden weights
    self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs))
    # bias of hidden units
    self.bias = np.random.random((1, self.numHiddenNeurons)) * 2 - 1
    # hidden to output layer connection
    self.beta = np.random.random((self.numHiddenNeurons, self.outputs))

    # auxiliary matrix used for sequential learning
    self.M = None

    self.BN=BN
    self.forgettingFactor = forgettingFactor

  def batchNormalization(self, H, scaleFactor=1, biasFactor=0):

    H_normalized = (H - H.mean()) / (np.sqrt(H.var() + 0.0001))
    H_normalized = scaleFactor * H_normalized + biasFactor

    return H_normalized

  def calculateHiddenLayerActivation(self, features):
    """
    Calculate activation level of the hidden layer
    :param features feature matrix with dimension (numSamples, numInputs)
    :return: activation level (numSamples, numHiddenNeurons)
    """
    if self.activationFunction is "sig":
      V = linear(features, self.inputWeights,self.bias)
      if self.BN:
        V = self.batchNormalization(V)
      H = sigmoidActFunc(V)
    else:
      print " Unknown activation function type"
      raise NotImplementedError

    return H
  def initializePhase(self, features, targets):
    """
    Step 1: Initialization phase
    :param features feature matrix with dimension (numSamples, numInputs)
    :param targets target matrix with dimension (numSamples, numOutputs)
    """
    assert features.shape[0] == targets.shape[0]
    assert features.shape[1] == self.inputs
    assert targets.shape[1] == self.outputs

    # randomly initialize the input->hidden connections
    self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs))
    self.inputWeights = self.inputWeights * 2 - 1

    if self.activationFunction is "sig":
      self.bias = np.random.random((1, self.numHiddenNeurons)) * 2 - 1
    else:
      print " Unknown activation function type"
      raise NotImplementedError

    H0 = self.calculateHiddenLayerActivation(features)
    self.M = pinv(np.dot(np.transpose(H0), H0))

    self.beta = np.dot(pinv(H0), targets)


  def train(self, features, targets):
    """
    Step 2: Sequential learning phase
    :param features feature matrix with dimension (numSamples, numInputs)
    :param targets target matrix with dimension (numSamples, numOutputs)
    """
    (numSamples, numOutputs) = targets.shape
    assert features.shape[0] == targets.shape[0]

    H = self.calculateHiddenLayerActivation(features)

    #H = features
    Ht = np.transpose(H)
    try:
      scale = 1 / (self.forgettingFactor)
      self.M = scale * self.M - np.dot(scale * self.M,
                                       np.dot(Ht, np.dot(
                                         pinv(np.eye(numSamples) + np.dot(H, np.dot(scale * self.M, Ht))),
                                         np.dot(H, scale * self.M))))
      #forgetting method 1
      #self.beta = (self.forgettingFactor) * self.beta + np.dot(self.M, np.dot(Ht, targets - np.dot(H, (self.forgettingFactor) * self.beta)))
      #forgetting method 2
      #self.beta = (self.forgettingFactor) * self.beta + (self.forgettingFactor)*np.dot(self.M, np.dot(Ht, targets - np.dot(H,self.beta)))
      #forgetting method 3
      self.beta = self.beta + np.dot(self.M, np.dot(Ht, targets - np.dot(H,self.beta)))


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
    #H = features

    prediction = np.dot(H, self.beta)
    return prediction

