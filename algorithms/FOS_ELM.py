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

def orthogonalization(Arr):
  [Q, S, _] = np.linalg.svd(Arr)
  tol = max(Arr.shape) * np.spacing(max(S))
  r = np.sum(S > tol)
  Q = Q[:, :r]

  return Q

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



class FOSELM(object):
  def __init__(self, inputs, outputs, numHiddenNeurons, activationFunction, BN=False,forgettingFactor=0.999, ORTH = False, VFF_RLS=False, ADAPT=False,sigma=1,minForget=0.5):

    self.activationFunction = activationFunction
    self.inputs = inputs
    self.outputs = outputs
    self.numHiddenNeurons = numHiddenNeurons

    # input to hidden weights
    self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs))
    self.ORTH = ORTH

    # bias of hidden units
    self.bias = np.random.random((1, self.numHiddenNeurons)) * 2 - 1
    # hidden to output layer connection
    self.beta = np.random.random((self.numHiddenNeurons, self.outputs))
    self.BN = BN
    # auxiliary matrix used for sequential learning
    self.M = None

    self.forgettingFactor = forgettingFactor
    self.trace=0
    self.thresReset=0.005

    # for VFF_RLS
    # parameters are set as recommended by Bodal et al.
    self.VFF_RLS=VFF_RLS
    if self.VFF_RLS:
      self.forgettingFactor=1
      self.gamma=pow(10,-3)
      self.upsilon=pow(10,-6)
      self.rho=0.99
    # for ADAPT
    self.ADAPT=ADAPT
    if ADAPT:
      self.sigma = sigma
      self.minForget=minForget
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


  def initializePhase(self, lamb=0.0001):
    """
    Step 1: Initialization phase
    :param features feature matrix with dimension (numSamples, numInputs)
    :param targets target matrix with dimension (numSamples, numOutputs)
    """

    # randomly initialize the input->hidden connections
    self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs))
    self.inputWeights = self.inputWeights * 2 - 1

    if self.ORTH:
      if self.numHiddenNeurons > self.inputs:
        self.inputWeights = orthogonalization(self.inputWeights)
      else:
        self.inputWeights = orthogonalization(self.inputWeights.transpose())
        self.inputWeights = self.inputWeights.transpose()

    if self.activationFunction is "sig":
      self.bias = np.random.random((1, self.numHiddenNeurons)) * 2 - 1
    else:
      print " Unknown activation function type"
      raise NotImplementedError

    self.M = inv(lamb*np.eye(self.numHiddenNeurons))
    self.beta = np.zeros([self.numHiddenNeurons,self.outputs])



  def train(self, features, targets,RLS=False,RESETTING=False):
    """
    Step 2: Sequential learning phase
    :param features feature matrix with dimension (numSamples, numInputs)
    :param targets target matrix with dimension (numSamples, numOutputs)
    """
    (numSamples, numOutputs) = targets.shape
    assert features.shape[0] == targets.shape[0]

    H = self.calculateHiddenLayerActivation(features)
    Ht = np.transpose(H)

    if RLS:

      if self.VFF_RLS:
      # suppose numSamples = 1

        # calculate output weight self.beta
        output = np.dot(H, self.beta)
        self.e = targets - output
        self.zeta = np.dot(H, np.dot(self.M, Ht))
        self.beta = self.beta + np.dot(np.dot(self.M, Ht), self.e) / (1 + self.zeta)

        # calculate covariance matrix self.M
        if self.zeta != 0:
          self.epsilon = self.forgettingFactor - (1 - self.forgettingFactor) / self.zeta
          self.M = self.M - np.dot(np.dot(self.M, Ht), np.dot(H, self.M)) / (1 / self.epsilon + self.zeta)

        # calculate forgetting factor self.forgettingFactor
        self.gamma = self.forgettingFactor * (self.gamma + pow(self.e, 2) / (1 + self.zeta))
        self.eta = pow(self.e, 2) / self.gamma
        self.upsilon = self.forgettingFactor * (self.upsilon + 1)
        self.forgettingFactor = 1 / (1 + (1 + self.rho) * (
        np.log(1 + self.zeta) + (((self.upsilon + 1) * self.eta / (1 + self.zeta + self.eta)) - 1) * (
        self.zeta / (1 + self.zeta))))

      elif self.ADAPT:
        self.RLS_k = np.dot(self.M, Ht)/(self.forgettingFactor + np.dot(H, np.dot(self.M, Ht)))
        self.RLS_e = targets - np.dot(H, self.beta)
        self.beta = self.beta + np.dot(self.RLS_k, self.RLS_e)
        self.forgettingFactor = 1 - (1 - np.dot(H,self.RLS_k))*pow(self.RLS_e,2)/self.sigma
        self.forgettingFactor= max(self.minForget,self.forgettingFactor)
        self.M = 1 / (self.forgettingFactor) * (self.M - np.dot(self.RLS_k, np.dot(H, self.M)))
        print self.forgettingFactor

      else:

        self.RLS_k = np.dot(np.dot(self.M,Ht),inv( self.forgettingFactor*np.eye(numSamples)+ np.dot(H,np.dot(self.M,Ht))))
        self.RLS_e = targets - np.dot(H,self.beta)
        self.beta = self.beta + np.dot(self.RLS_k,self.RLS_e)
        self.M = 1/(self.forgettingFactor)*(self.M - np.dot(self.RLS_k,np.dot(H,self.M)))

    else:

      scale = 1 / (self.forgettingFactor)
      self.M = scale * self.M - np.dot(scale * self.M,
                                       np.dot(Ht, np.dot(
                                         pinv(np.eye(numSamples) + np.dot(H, np.dot(scale * self.M, Ht))),
                                         np.dot(H, scale * self.M))))

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


      #self.beta = (self.forgettingFactor)*self.beta + np.dot(self.M, np.dot(Ht, targets - np.dot(H, (self.forgettingFactor)*self.beta)))
      #self.beta = (self.forgettingFactor)*self.beta + (self.forgettingFactor)*np.dot(self.M, np.dot(Ht, targets - np.dot(H, self.beta)))
      self.beta = self.beta + np.dot(self.M, np.dot(Ht, targets - np.dot(H, self.beta)))


  def predict(self, features):
    """
    Make prediction with feature matrix
    :param features: feature matrix with dimension (numSamples, numInputs)
    :return: predictions with dimension (numSamples, numOutputs)
    """
    H = self.calculateHiddenLayerActivation(features)
    prediction = np.dot(H, self.beta)
    return prediction

