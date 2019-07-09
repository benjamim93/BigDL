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

package com.intel.analytics.bigdl.example.languagemodel

import java.awt.geom.Point2D

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.text.{LabeledSentenceToSample, _}
import com.intel.analytics.bigdl.dataset.{DataSet, SampleToMiniBatch}
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module, TimeDistributedCriterion}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.utils.{Engine, MklDnn}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.example.languagemodel.Utils._
import com.intel.analytics.bigdl.models.rnn.SequencePreprocess
import com.intel.analytics.bigdl.tensor.Tensor

object Test {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.example").setLevel(Level.INFO)
  val logger = Logger.getLogger(getClass)
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
        // .transform(SampleToMiniBatch[Float](param.batchSize))
        .toDistributed()
        .data(train = false)

//      val validationSet = DataSet.rdd(sc.parallelize(
//        SequencePreprocess.reader(validData, param.numSteps)))
//        .transform(TextToLabeledSentence[Float](param.numSteps))
//        .transform(LabeledSentenceToSample[Float](
//          oneHot = false,
//          fixDataLength = None,
//          fixLabelLength = None))
//        .transform(SampleToMiniBatch[Float](param.batchSize))

//      val testSet = DataSet.rdd(sc.parallelize(
//        SequencePreprocess.reader(testData, param.numSteps)))
//        .transform(TextToLabeledSentence[Float](param.numSteps))
//        .transform(LabeledSentenceToSample[Float](
//          oneHot = false,
//          fixDataLength = None,
//          fixLabelLength = None))
//        .transform(SampleToMiniBatch[Float](param.batchSize))

      // val model = Module.load[Float](param.modelSnapshot.get)
      val model = PTBModel.lstm(
        inputSize = param.vocabSize,
        hiddenSize = param.hiddenSize,
        outputSize = param.vocabSize,
        numLayers = param.numLayers,
        keepProb = param.keepProb)

      var start = System.nanoTime()
      val resultdnn = model.evaluate(trainSet, Array(new Loss[Float](
        TimeDistributedCriterion[Float](
          CrossEntropyCriterion[Float](),
          sizeAverage = false, dimension = 1))), batchSize = Some(param.batchSize))
      resultdnn.foreach(r => println(s"${r._2} is ${r._1}"))
      var end = System.nanoTime()
      println(s"Time consumed: ${(end - start) * 1e-9}")
      println(s"Throughput: ${param.batchSize.toDouble / (end - start) * 1e9}")

      sc.stop()
    })
  }
}





