# -*- coding: utf-8 -*-

import numpy as np

class levelHV():
    def __init__(self, minVal, maxVal, nLevels=100, dim=10000):
        self.base = generateHV(dim);        #HV for minVal
        self.channelHV = generateHV(dim); 
        self.minVal = minVal; 
        self.maxVal = maxVal; 
        self.nLevels = nLevels;
        self.dim = dim; 
        self.levelHVs = self.getLevelHVs(); 

    def generateHV(N=1000, K=0.5):
        arr = np.zeros(N, dtype=int);
        sparsity = int(K*N); 
        arr[:sparsity]  = 1; 
        np.random.shuffle(arr); 
        return arr  
    
    def getLevelHVs(self): 
        levelHVs = [self.base]; 
        currentHV = self.base; 
        nFlipsPerLevel = self.dim/self.nLevels; 
        #Vector bitFlipPositions is used to flip the bits 
        bitFlipPositions = np.zeros(self.dim, dtype=int); 
        bitFlipPositions[:nFlipsPerLevel]  = 1;
        for i in range(1, self.nLevels):
            np.random.shuffle(bitFlipPositions);
            #XOR flips bits at positions where bitFlipPositions is 1
            newHV = np.bitwise_xor(currentHV, bitFlipPositions); 
            levelHVs.append(newHV); 
            currentHV = newHV; 
        return levelHVs
            
    def rangeNorm(self, value): 
        r = (value-self.minVal)/float(self.maxVal-self.minVal);
        r = np.clip(r, 0, 1); 
        return r
        
    def getHV(self, value):
        #Find which level
        r = self.rangeNorm(value);   
        level = r*(self.nLevels-1);
        level = int(round(level)); 
        levelHV = self.levelHVs[level];
        out = np.bitwise_xor(self.channelHV, levelHV);      #bind
        return out


def getSampleHV(sample, sensorLevelHVs): 
    sensorHVs = []; 
    for i in range(0, len(sample)):
        hv = sensorLevelHVs[i].getHV(sample[i]); 
        sensorHVs.append(hv); 
    sampleHV = csum(sensorHVs); 
    return sampleHV

def generateTrainingHVs(trainingData, trainingDataLabels):
    nSensors = len(trainingData[0]); 
    sensorLevelHVs = [];        
    sampleHVs = {}; 
    minVal = np.min(trainingData)
    maxVal = np.max(trainingData) 
    #Find HBVs for each sensor
    for i in range(0, nSensors):
        #sensorVals = trainingData[:, i]; 
        #minVal, maxVal = min(sensorVals), max(sensorVals); 
        sensorLevelHV = levelHV(minVal, maxVal); 
        sensorLevelHVs.append(sensorLevelHV);
    #Find HBV for each training sample
    for i in range(0, len(trainingData)): 
        sampleHV = getSampleHV(trainingData[i], sensorLevelHVs);
        classLabel = trainingDataLabels[i]; 
        if classLabel in sampleHVs.keys(): 
            sampleHVs[classLabel].append(sampleHV); 
        else: 
            sampleHVs[classLabel] = [sampleHV]; 
    #Find HBV for each training class
    trainingHVs= [];            #samples from each class are superimposed and stored here
    for i in trainingDataLabels: 
        trainingHVs.append(csum(sampleHVs[i])); 
    return trainingHVs, sensorLevelHVs

def classifyTestData(testData, testDataLabels, trainingDataLabels, trainingHVs, sensorLevelHVs): 
    testHVs = []; 
    predictions = []; 
    sMatrix = []; 
    nCorrect = 0; 
    for i in range(0, len(testData)): 
        testHV = getSampleHV(testData[i], sensorLevelHVs);
        p, similarities = findBestMatch(testHV, trainingHVs); 
        predictions.append(p); 
        sMatrix.append(similarities); 
        testHVs.append(testHV); 
        if(trainingDataLabels[p]==testDataLabels[i]): 
            nCorrect+=1; 
    return nCorrect, predictions, sMatrix, testHVs

def generateHV(d=1000, k=0.5):
    arr = np.zeros(d, dtype=int);
    sparsity = int(k*d); 
    arr[:sparsity] = 1; 
    np.random.shuffle(arr); 
    return arr                

def xor(a, b): 
    out = np.bitwise_xor(a, b); 
    return out

def csumThreshold(vectorElement, theta):
    if vectorElement > theta: 
        return 1
    elif vectorElement < theta: 
        return 0
    elif vectorElement == theta:
        if(np.random.random()>0.5):
            #return 0
            return 1
        else:
            return 0

def csum(vlist):        #consensum sum; vlist is list of HD vectors
    theta = len(vlist)/2.0;     
    dim = len(vlist[0]); 
    sum1 = np.zeros(dim); 
    for i in vlist: 
        sum1 = sum1 + i; 
    vfunc = np.vectorize(csumThreshold); 
    out = vfunc(sum1, theta); 
    return out

def hammingS(a, b):
    nBits = len(a); 
    nNonMatches = np.bitwise_xor(a, b); 
    nMatches = nBits - sum(nNonMatches); 
    d = nMatches/float(nBits); 
    return d; 
    #return np.count_nonzero(a!=b)/float(len(a)); 

def findBestMatch(testHV, trainingHVs):         #can be vectorized if needed 
    hammingSimilarities = []; 
    for i in trainingHVs:
        hammingSimilarities.append(hammingS(testHV, i));
    bestMatch = hammingSimilarities.index(max(hammingSimilarities)); 
    return bestMatch, hammingSimilarities