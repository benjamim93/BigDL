/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn.mkldnn.models

import com.intel.analytics.bigdl.dataset.text.{LabeledSentenceToSample, TextToLabeledSentence}
import com.intel.analytics.bigdl.dataset.{DataSet, SampleToMiniBatch}
import com.intel.analytics.bigdl.mkl.{AlgKind, Direction, Memory}
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.mkldnn._
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.example.languagemodel.Utils.{TrainParams, trainParser}
import com.intel.analytics.bigdl.models.rnn.SequencePreprocess
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.optim._

object PTBModel {
  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val conf = Engine.createSparkConf()
        .setAppName("Train ptbModel on text")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val (trainData, validData, testData, dictionary) = SequencePreprocess(
        param.dataFolder, param.vocabSize)

      val trainSet = DataSet.rdd(sc.parallelize(
        SequencePreprocess.reader(trainData, param.numSteps)))
        .transform(TextToLabeledSentence[Float](param.numSteps))
        .transform(LabeledSentenceToSample[Float](
          oneHot = false,
          fixDataLength = None,
          fixLabelLength = None))
        .transform(SampleToMiniBatch[Float](param.batchSize))

      val validationSet = DataSet.rdd(sc.parallelize(
        SequencePreprocess.reader(validData, param.numSteps)))
        .transform(TextToLabeledSentence[Float](param.numSteps))
        .transform(LabeledSentenceToSample[Float](
          oneHot = false,
          fixDataLength = None,
          fixLabelLength = None))
        .transform(SampleToMiniBatch[Float](param.batchSize))

      val model = lstm(
        inputSize = param.vocabSize,
        hiddenSize = param.hiddenSize,
        outputSize = param.vocabSize,
        numSteps = param.numSteps,
        batchSize = param.batchSize,
        numLayers = param.numLayers,
        keepProb = param.keepProb
      )

      model.compile(TrainingPhase)

      val optimMethod = new Adagrad[Float](learningRate = param.learningRate,
        learningRateDecay = param.learningRateDecay)

      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = nn.TimeDistributedCriterion[Float](
          nn.CrossEntropyCriterion[Float](), sizeAverage = false, dimension = 1)
      )

      optimizer
        .setValidation(Trigger.everyEpoch, validationSet, Array(new Loss[Float](
          nn.TimeDistributedCriterion[Float](
            nn.CrossEntropyCriterion[Float](),
            sizeAverage = false, dimension = 1))))
        .setOptimMethod(optimMethod)
        .setEndWhen(Trigger.maxEpoch(param.nEpochs))
        .optimize()
      sc.stop()
    })
  }

  def lstm(
    inputSize: Int,
    hiddenSize: Int,
    outputSize: Int,
    numSteps: Int,
    batchSize: Int,
    numLayers: Int,
    keepProb: Float = 2.0f)
  : DnnGraph = {
    val input = Input(Array(batchSize, numSteps, inputSize), Memory.Format.ntc).inputs()

    val embeddingLookup = BlasWrapper(nn.LookupTable[Float](inputSize, hiddenSize)).inputs(input)

    val inputs = if (keepProb < 1) {
      Dropout(keepProb).inputs((embeddingLookup))
    } else embeddingLookup

    val shapeNTC = Array(batchSize, numSteps, inputSize)
    val shapeTNC = Array(numSteps, batchSize, inputSize)

    val ntc2tnc = ReorderMemory(
      inputFormat = HeapData(shapeNTC, Memory.Format.ntc),
      outputFormat = HeapData(shapeTNC, Memory.Format.tnc),
      gradInputFormat = HeapData(shapeTNC, Memory.Format.tnc),
      gradOutputFomat = HeapData(shapeNTC, Memory.Format.ntc)
    ).inputs(inputs)

    val lstm = RNN(
      mode = AlgKind.VanillaLstm,
      inputSize = inputSize,
      hiddenSize = hiddenSize,
      f = AlgKind.EltwiseTanh,
      direction = Direction.UnidirectionalLeft2Right,
      layers = numLayers
    ).inputs(ntc2tnc)

    val tnc2ntc = ReorderMemory(
      inputFormat = HeapData(shapeTNC, Memory.Format.tnc),
      outputFormat = HeapData(shapeNTC, Memory.Format.ntc),
      gradInputFormat = HeapData(shapeNTC, Memory.Format.ntc),
      gradOutputFomat = HeapData(shapeTNC, Memory.Format.tnc)
    ).inputs(lstm)

    val output = Linear(hiddenSize, outputSize).inputs(tnc2ntc)

    DnnGraph(Seq(input), Seq(output))
  }
}