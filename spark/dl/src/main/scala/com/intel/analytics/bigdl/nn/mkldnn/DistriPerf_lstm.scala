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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.example.languagemodel.PTBModel
import com.intel.analytics.bigdl.example.languagemodel.PTBModel.addLayer
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils._
import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import scopt.OptionParser
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}

import scala.collection.mutable.ArrayBuffer
import com.intel.analytics.bigdl.mkl.AlgKind
import com.intel.analytics.bigdl.mkl.Direction
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase

object DistriPerf_lstm {
  val logger = Logger.getLogger(getClass)

  val parser = new OptionParser[DistriPerflstmParams]("BigDL w/ Dnn Local Model Performance Test") {
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
    opt[Int]("numLayers")
      .text("The num of layers")
      .action((v, p) => p.copy(numLayers = v))
    opt[Int]('x', "threadNum")
      .text("Number of threads")
      .action((v, p) => p.copy(threadNum = v))
    opt[String]("engineType")
      .text("Engine type")
      .action((v, p) => p.copy(engineType = v))
  }

  def getTopTimes(times: Array[(AbstractModule[_ <: Activity, _ <: Activity, Float],
    Long, Long)], allSum: Long): Unit = {
    var forwardSum = 0L
    var backwardSum = 0L
    times.foreach(x => {
      forwardSum += x._2
      backwardSum += x._3
    })
    println(s"forwardSum = ${forwardSum/1e9} realTime ${allSum/1e9} backwardSum = ${backwardSum/1e9}")

    val timeBuffer = new ArrayBuffer[(AbstractModule[_ <: Activity,
      _ <: Activity, Float], Long)]
    var i = 0
    while (i < times.length) {
      val rate = times(i)._2.toDouble/ allSum
      timeBuffer.append((times(i)._1, times(i)._2))
      i += 1
    }
    val sortData = timeBuffer.sortBy(a => a._2)
    var m = 0
    while (m < sortData.length) {
      val layer = sortData(m)._1.getName()
      println(layer + "__" + sortData(m)._1 + "__" + (sortData(m)._2/1e9))
      m += 1
    }
  }

  def blasPredict(params: DistriPerflstmParams, sc: SparkContext, miniBatch: MiniBatch[Float]):
  Unit = {
    val subModelNumber = Engine.coreNumber()

    val input = nn.Input[Float]()
    val output = nn.Recurrent().add(nn.LSTM(params.hiddenSize, params.hiddenSize)).inputs(input)
    val model = nn.Graph(input, output)

    val workingModels = if (subModelNumber != 1) {
      val wb = Util.getAndClearWeightBias(model.parameters())
      val models = (1 to subModelNumber).map(i => {
        logger.info(s"Clone $i model...")
        val m = model.cloneModule()
        Util.putWeightBias(wb, m)
        m
      }).toArray
      Util.putWeightBias(wb, model)
      models
    } else {
      Array(model)
    }

    val inputSize = miniBatch.size()
    val stackSize = miniBatch.size() / subModelNumber
    val extraSize = miniBatch.size() % subModelNumber
    val parallelism = if (stackSize == 0) extraSize else subModelNumber
    val inputBuffer = new Array[MiniBatch[Float]](parallelism)
    var b = 0
    while (b < parallelism) {
      val offset = b * stackSize + math.min(b, extraSize) + 1
      val length = stackSize + (if (b < extraSize) 1 else 0)
      inputBuffer(b) = miniBatch.slice(offset, length)
      b += 1
    }

    // warm up
    println(s"engine default pool size ${Engine.default.getPoolSize}")
    val warmup = 20
    val warmpResults = Engine.default.invoke((0 until subModelNumber).map(i =>
      () => {
        val localModel = workingModels(i)
        val data = inputBuffer(i)
        for (w <- 0 to warmup) {
          localModel.forward(data.getInput())
          localModel.backward(data.getInput(), data.getTarget())
        }
        1
      }))
    Engine.default.sync(warmpResults)

    println(s"start predict throughput test [Uni L2R ${params.numLayers} Layer(s)]: ")
    val start = System.nanoTime()
    for (it <- 0 to params.iteration) {
      val results = Engine.default.invoke((0 until subModelNumber).map(i =>
        () => {
          val localModel = workingModels(i)
          val data = inputBuffer(i)
          // val s1 = System.nanoTime()
          localModel.forward(data.getInput())
          localModel.backward(data.getInput(), data.getTarget())
//          val e1 = System.nanoTime() - s1
//          getTopTimes(localModel.getTimes(), e1)
//          println(s"iteration time ${e1/1e9}")
//          println('\n')
//          localModel.resetTimes()
          1
        }))
      Engine.default.sync(results)
    }
    val end = System.nanoTime()

    logger.info(s"[Uni L2R ${params.numLayers} Layer(s)] result: ")
    logger.info(s"Average Throughput" +
      s"is ${params.batchSize.toDouble * params.iteration / (end - start) * 1e9}" +
      s" records / second")
  }

  def dnnPredict(params: DistriPerflstmParams, sc: SparkContext, miniBatch: MiniBatch[Float]):
  Unit = {
    val inputShape = Array(params.seqLength, params.batchSize, params.hiddenSize)
    val outputShape = Array(params.seqLength, params.batchSize, params.hiddenSize)

    val f = AlgKind.EltwiseTanh
    var direction = Direction.UnidirectionalLeft2Right

    val input = Input(inputShape, Memory.Format.tnc).inputs()
    val lstm = RNN(AlgKind.VanillaLstm, params.hiddenSize,
      params.hiddenSize, f, direction, layers = params.numLayers).inputs(input)
    val mkldnn_model = DnnGraph(Seq(input), Seq(lstm))

    println(s"\nstart predict throughput test [Uni L2R ${params.numLayers} Layer(s)]: ")

    val start = System.nanoTime()
    Engine.dnnComputing.invokeAndWait2((0 until 1).map(i => () => {
      mkldnn_model.compile(TrainingPhase)

      for (i <- 1 to params.iteration) {
        mkldnn_model.forward(miniBatch.getInput())
        mkldnn_model.backward(miniBatch.getInput(), miniBatch.getTarget())
      }
    }))
    val end = System.nanoTime()

    logger.info(s"[Uni L2R ${params.numLayers} Layer(s)] result: ")
    logger.info(s"Average Throughput" +
      s"is ${params.batchSize.toDouble * params.iteration / (end - start) * 1e9}" +
      s" records / second.")
  }

  def main(argv: Array[String]): Unit = {
    parser.parse(argv, new DistriPerflstmParams()).foreach { params =>
      println("batchSize = " + params.batchSize)
      println("iterations = " + params.iteration)
      println("seqLength = " + params.seqLength)
      println("inputSize = " + params.inputSize)
      println("hiddenSize = " + params.hiddenSize)
      println("outputSize = " + params.outputSize)
      println("numLayers = " + params.numLayers)
      println("engineType = " + params.engineType)

      if (params.engineType == "mkldnn") {
        System.setProperty("bigdl.engineType", "mkldnn")
        System.setProperty("bigdl.mklNumThreads", params.threadNum.toString)
      }

      if (params.engineType == "blas") {
        System.setProperty("bigdl.engineType", "mklblas")
      }

      val conf = Engine.createSparkConf()
        .setAppName("Test perf")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)

      Engine.init
      Engine.setNodeAndCore(Engine.nodeNumber(), Engine.coreNumber())

      RNG.setSeed(100)
      val minibatch = if (params.engineType == "mkldnn") {
        val TNCshape = Array(params.seqLength, params.batchSize, params.hiddenSize)
        MiniBatch(Tensor[Float](TNCshape).rand(), Tensor[Float](TNCshape).rand())
      }
      else {
        val NTCshape = Array(params.batchSize, params.seqLength, params.hiddenSize)
        MiniBatch(Tensor[Float](NTCshape).rand(), Tensor[Float](NTCshape).rand())
      }

      if (params.engineType == "mkldnn") {
        dnnPredict(params, sc, minibatch)
      }

      if (params.engineType == "blas") {
        blasPredict(params, sc, minibatch)
      }
    }
  }
}

case class DistriPerflstmParams (
  batchSize: Int = 20,
  iteration: Int = 100,
  seqLength: Int = 300,
  inputSize: Int = 800,
  hiddenSize: Int = 800,
  outputSize: Int = 800,
  numLayers: Int = 1,
  threadNum: Int = 1,
  engineType: String = "mkldnn",
  model: String = "vanilla_lstm",
  threadPredict: Boolean = true
)
