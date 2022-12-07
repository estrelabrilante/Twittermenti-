//
//  main.swift
//  TwitterSentimentModelMaker
//
//  Created by user215496 on 12/3/22.
//

import Cocoa
import CreateML

//create machinelearning model
let data = try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/user215496/Downloads/twitter-sanders-apple3.csv"))
                                           
//training data and testing data
let (trainingData, testingData) = data.randomSplit(by: 0.8,seed: 5)


let sentimentClassifier = try MLTextClassifier(trainingData: trainingData, textColumn: "text", labelColumn: "class")

let evaluationMetrics = sentimentClassifier.evaluation(on: testingData, textColumn: "text", labelColumn: "class")
let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100
let metadata = MLModelMetadata(author: "Shynu",shortDescription: "A model trained to classify sentiment on Twitter",version: "1.0")

try sentimentClassifier.write(to: URL(fileURLWithPath: "/Users/user215496/Downloads/TwitterSentimentClassifier.mlmodel"))
//testing

try sentimentClassifier.prediction(from: "Apple is a terrible company")

try sentimentClassifier.prediction(from: "Apple has workholic workers")






