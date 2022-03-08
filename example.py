#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import hd_lib as hd
import numpy as np
import misc_lib as misc

np.random.seed(1); 

dataset = np.load('./data.npz');    #training samples (one-shot) from one plume, test samples from different plume
trainingOdors = dataset['trainingSet'];
trainingOdorLabels = dataset['trainingSetLabels']; 
testOdors = dataset['testSet']; 
testOdorLabels = dataset['testSetLabels']; 

nOdors = len(trainingOdors); 
print("Number of odors to train = " + str(len(trainingOdors))); 
print("Number of odors to test = " + str(len(testOdors))); 

#Generate HD representations of training and test data; then classify
trainingHVs, sensorLevelHVs = hd.generateTrainingHVs(trainingOdors, trainingOdorLabels); 
nCorrect, pValues, similarities, testHVs = hd.classifyTestData(testOdors, testOdorLabels, trainingOdorLabels, trainingHVs, sensorLevelHVs); 
fractionCorrect = float(nCorrect)/len(testOdors); 
print("Classification Performance HD: " + str(fractionCorrect)); 

#Below is 1-NN classification;                            
nCorrectNN, pValuesNN, similaritiesNN = misc.classify(testOdors, testOdorLabels, trainingOdors, trainingOdorLabels);
fractionCorrectNN = float(nCorrectNN)/len(testOdors); 
print("Classification Performance NN: " + str(fractionCorrectNN)); 

