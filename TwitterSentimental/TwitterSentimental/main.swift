//
//  main.swift
//  TwitterSentimental
//
//  Created by SHYNU MARY VARGHESE on 2023-02-07.
//

import Foundation
import Cocoa
import CreateML
let data = try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/shynumaryvarghese/Downloads/twitter-sanders-apple3.csv"))
//randomly splitting 80% training data and 20% tesing data
let(trainingData, testingData) = data.randomSplit(by: 0.8, seed: 5)
let sentimentClassifier = try MLTextClassifier(trainingData: trainingData, textColumn: "text", labelColumn: "class")
//evaluationMetrics
let evaluationMetrics = sentimentClassifier.evaluation(on: testingData, textColumn: "text", labelColumn:"class")
//accuracy
let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100
let metadata = MLModelMetadata(author: "Shynu", shortDescription: "A model trained to classify sentiment on Tweets", version: "1.0")
try sentimentClassifier.write(to: URL(fileURLWithPath: "Users/shynumaryvarghese/Downloads/twitter-sanders-apple3.mlmodel"))

try sentimentClassifier.prediction(from: "@Apple is a terrible company")

//print("Hello, World!")

