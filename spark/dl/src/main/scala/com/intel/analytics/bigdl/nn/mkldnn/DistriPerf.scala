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

// package com.intel.analytics.bigdl.utils.intermediate
package com.intel.analytics.bigdl.nn.mkldnn

import java.util.concurrent.atomic.AtomicInteger

import breeze.linalg.*
import breeze.numerics._
import com.intel.analytics.bigdl.{Module, utils}
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.example.languagemodel.PTBModel
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.mkl.{AlgKind, Direction, Memory}
import com.intel.analytics.bigdl.models.inception.Inception_v1_NoAuxClassifier
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
// import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.{Graph, Module, StaticGraph}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.Recurrent
// import com.intel.analytics.bigdl.nn.mkldnn.{Phase}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase
import com.intel.analytics.bigdl.nn.mkldnn.models.Vgg_16
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.{DenseTensorMath, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils._
import org.apache.log4j.Logger
import org.apache.spark.{SparkContext, broadcast}
import scopt.OptionParser
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

import com.intel.analytics.bigdl.nn

object DistriPerf {
  val logger = Logger.getLogger(getClass)

  val parser = new OptionParser[DistriPerfParams]("BigDL w/ Dnn Local Model Performance Test") {
    opt[Int]('b', "batchSize")
      .text("Batch size of input data")
      .action((v, p) => p.copy(batchSize = v))
    opt[Int]('i', "iteration")
      .text("Iteration of perf test. The result will be average of each iteration time cost")
      .action((v, p) => p.copy(iteration = v))
    opt[Int]('s', "seqLength")
      .text("The length of sequence")
      .action((v, p) => p.copy(seqLength = v))
    opt[Int]("inputSize")
      .text("The size of input")
      .action((v, p) => p.copy(inputSize = v))
    opt[Int]("hiddenSize")
      .text("The size of hidden state")
      .action((v, p) => p.copy(hiddenSize = v))
    opt[Int]("outputSize")
      .text("The size of output")
      .action((v, p) => p.copy(outputSize = v))
    opt[Int]('x', "threadNum")
      .text("Number of threads")
      .action((v, p) => p.copy(threadNum = v))
  }

  def predict(params: DistriPerfParams, sc: SparkContext): Unit = {
    println("batchSize = " + params.batchSize)
    println("iterations = " + params.iteration)
    println("seqLength = " + params.seqLength)
    println("inputSize = " + params.inputSize)
    println("hiddenSize = " + params.hiddenSize)
    println("outputSize = " + params.outputSize)

//    val inputShape = Array(params.batchSize, params.seqLength)
//    val outputShape = Array(params.batchSize, params.seqLength, params.outputSize)

    val model = PTBModel.lstm(
      inputSize = params.inputSize,
      hiddenSize = params.hiddenSize,
      outputSize = params.outputSize,
      numLayers = 1)

    model.asInstanceOf[StaticGraph[Float]]
      .setInputFormats(Seq(Memory.Format.nc))
      .setOutputFormats(Seq(Memory.Format.ntc))

    println("\nstart predict throughput test [Uni L2R 1 Layer]: ")

    val input = Tensor[Float](Array(params.batchSize, params.seqLength))
    val gradOutput = Tensor[Float](Array(params.batchSize, params.seqLength, params.outputSize))

    val start = System.nanoTime()
    for (i <- 1 to params.iteration) {
      model.forward(input).toTensor
      model.backward(input, gradOutput).toTensor

    }
    val end = System.nanoTime()

    logger.info("[Uni L2R 1 Layer] result: ")
    logger.info(s"Average Throughput" +
      s"is ${params.batchSize.toDouble * params.iteration / (end - start) * 1e9}" +
      s" record / second.")
  }

  def main(argv: Array[String]): Unit = {
    parser.parse(argv, new DistriPerfParams()).foreach { params =>
      System.setProperty("bigdl.engineType", "mkldnn")
      System.setProperty("bigdl.mklNumThreads", params.threadNum.toString)

      val conf = Engine.createSparkConf()
        .setAppName("Test perf")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)

      Engine.init
      Engine.setNodeAndCore(Engine.nodeNumber(), Engine.coreNumber())

      println("Thread number: " + params.threadNum)

      predict(params, sc)
    }
  }
}

case class DistriPerfParams (
  batchSize: Int = 20,
  iteration: Int = 100,
  seqLength: Int = 300,
  inputSize: Int = 800,
  hiddenSize: Int = 800,
  outputSize: Int = 800,
  threadNum: Int = 1,
  model: String = "vanilla_lstm",
  threadPredict: Boolean = true
)

