#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np

def addGaussianNoise(data, dataLabels, noiseLevel=1.0, nPerData=10): 
    noisyData = []; 
    noisyDataLabel = []; 
    for i in range(0, len(data)): 
        for j in range(0, nPerData):
            sampleID = i*nPerData + j; 
            noisyData.append([]);
            noisyDataLabel.append(dataLabels[i]); 
            for k in range(0, len(data[i])): 
                noise = np.random.normal(0, noiseLevel); 
                noisyData[sampleID].append(data[i][k] + noise);
    return noisyData, noisyDataLabel
    
def euclideanDistance(sample1, sample2):
    s1 = np.asarray(sample1); 
    s2 = np.asarray(sample2); 
    d = np.linalg.norm(s2-s1); 
    return d

def cosSimilarity(sample1, sample2): 
    dotP = np.dot(sample1, sample2); 
    normP = np.linalg.norm(sample1)*np.linalg.norm(sample2); 
    s = dotP/float(normP); 
    return s; 

def findSimilarity(sample1, sample2): 
    #s = cosSimilarity(sample1, sample2); 
    s = euclideanDistance(sample1, sample2); 
    return s; 

def findSImatrix(testSet, trainingSet): 
    SImatrix = np.zeros((len(testSet), len(trainingSet))); 
    for i in range(0, len(testSet)): 
        for j in range(0, len(trainingSet)): 
            s = findSimilarity(testSet[i], trainingSet[j]); 
            SImatrix[i][j] = s; 
    return SImatrix

def classify(testSet, testSetLabels, trainingSet, trainingSetLabels): 
    SImatrix = findSImatrix(testSet, trainingSet); 
    pValues = []; 
    nCorrect = 0; 
    for i in range(0, len(SImatrix)):
        #bestMatchID = np.argmax(SImatrix[i, :]); 
        bestMatchID = np.argmin(SImatrix[i, :]); 
        pValues.append(bestMatchID); 
        if(trainingSetLabels[bestMatchID]==testSetLabels[i]):
            nCorrect+=1; 
    return nCorrect, pValues, SImatrix