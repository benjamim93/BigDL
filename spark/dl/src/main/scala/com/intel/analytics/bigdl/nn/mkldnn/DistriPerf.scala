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
import com.intel.analytics.bigdl.mkl.{AlgKind, Direction, Memory}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{DnnTensor, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils._
import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import scopt.OptionParser
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.utils.intermediate.IRGraph

import scala.collection.mutable.ArrayBuffer

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
    opt[Int]("vocabSize")
      .text("The size of vocabulary")
      .action((v, p) => p.copy(vocabSize = v))
    opt[Int]("hiddenSize")
      .text("The size of hidden state")
      .action((v, p) => p.copy(hiddenSize = v))
    opt[Int]("numLayers")
      .text("The num of layers")
      .action((v, p) => p.copy(numLayers = v))
    opt[Int]('x', "threadNum")
      .text("Number of threads")
      .action((v, p) => p.copy(threadNum = v))
    opt[String]("engineType")
      .text("Engine type")
      .action((v, p) => p.copy(engineType = v))
    opt[Boolean]("fused")
      .text("fused")
      .action((v, p) => p.copy(fused = v))
    opt[Boolean]("verbose")
      .text("verbose")
      .action((v, p) => p.copy(verbose = v))
    opt[Boolean]("infer")
      .text("infer")
      .action((v, p) => p.copy(infer = v))
  }

  def getTopTimes(times: Array[(AbstractModule[_ <: Activity, _ <: Activity, Float],
    Long, Long)] // , allSum: Long
    ): Unit = {
    var forwardSum = 0L
    var backwardSum = 0L
    times.foreach(x => {
      forwardSum += x._2
      backwardSum += x._3
    })
    // println(s"forwardSum = ${forwardSum/1e9} realTime ${allSum/1e9} backwardSum = ${backwardSum/1e9}")

    val timeBuffer = new ArrayBuffer[(AbstractModule[_ <: Activity,
      _ <: Activity, Float], Long)]
    var i = 0
    while (i < times.length) {
      // val rate = times(i)._2.toDouble/ allSum
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

  def blasPredict(params: DistriPerfParams, sc: SparkContext, miniBatch: MiniBatch[Float]): Unit = {
    val subModelNumber = Engine.coreNumber()

    val model = PTBModel.lstm(
      params.vocabSize,
      params.hiddenSize,
      params.vocabSize,
      params.numLayers)

    val workingModels = if (subModelNumber != 1) {
      val wb = Util.getAndClearWeightBias(model.parameters())
      val models = (1 to subModelNumber).map(i => {
        logger.info(s"Clone $i model...")
        val m = model.cloneModule()
        Util.putWeightBias(wb, m)
        m.asInstanceOf[nn.Graph[Float]]
          .modules(1)
          .asInstanceOf[nn.LookupTable[Float]]
          .gradWeight = Tensor[Float](params.vocabSize, params.hiddenSize).zero()
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
    val warmup = 30
    val warmpResults = Engine.default.invoke((0 until subModelNumber).map(i =>
      () => {
        val localModel = workingModels(i)
        val data = inputBuffer(i)
        for (w <- 0 to warmup) {
          localModel.forward(data.getInput())
          if (!params.infer) {
            localModel.backward(data.getInput(), data.getTarget())
          }
        }
        1
      }))
    Engine.default.sync(warmpResults)

    println(s"start predict throughput test [PTBModel-LSTM BLAS]: ")
    val start = System.nanoTime()
    for (it <- 0 to params.iteration) {
      val results = Engine.default.invoke((0 until subModelNumber).map(i =>
        () => {
          val localModel = workingModels(i)
          val data = inputBuffer(i)
          if (params.verbose) {
            // val s1 = System.nanoTime()
            localModel.forward(data.getInput())
            if (!params.infer) {
              localModel.backward(data.getInput(), data.getTarget())
            }
            // val e1 = System.nanoTime()
            getTopTimes(localModel.getTimes())
            // println(s"iteration time ${(e1 - s1)/1e9}")
            println('\n')
            localModel.resetTimes()
          }
          else {
            localModel.forward(data.getInput())
            if (!params.infer) {
              localModel.backward(data.getInput(), data.getTarget())
            }
          }
          1
        }))
      Engine.default.sync(results)
    }
    val end = System.nanoTime()

    println(s"Avg iteration time: ${(end - start)/(params.iteration * 1e9) } sec(s)")

    logger.info(s"[PTBModel-LSTM BLAS] result: ")
    logger.info(s"Average Throughput" +
      s"is ${params.batchSize.toDouble * params.iteration / (end - start) * 1e9}" +
      s" records / second")
  }

  def dnnPredict(params: DistriPerfParams, sc: SparkContext, miniBatch: MiniBatch[Float]): Unit = {
    val model = if (!params.fused) {
        PTBModel.lstm(
        inputSize = params.vocabSize,
        hiddenSize = params.hiddenSize,
        outputSize = params.vocabSize,
        numLayers = params.numLayers)
        .asInstanceOf[nn.StaticGraph[Float]]
        .setInputFormats(Seq(Memory.Format.nc))
        .setOutputFormats(Seq(Memory.Format.ntc))
        .toIRgraph()
    }

    else {
      val f = AlgKind.EltwiseTanh
      var direction = Direction.UnidirectionalLeft2Right

      val input = Input(Array(params.batchSize, params.seqLength), Memory.Format.nc).inputs()
      val embeddingLookup = BlasWrapper(nn.LookupTable[Float](params.vocabSize, params.hiddenSize))
        .inputs(input)
      val lstm = RNN(AlgKind.VanillaLstm, params.hiddenSize,
        params.hiddenSize, f, direction, layers = params.numLayers).inputs(embeddingLookup)
      val linear = nn.Linear(params.hiddenSize, params.vocabSize)
      val output = BlasWrapper(nn.TimeDistributed[Float](linear)).inputs(lstm)

      DnnGraph(Seq(input), Seq(output))
    }

    println(s"\nstart predict throughput test [PTBModel-LSTM MKLDNN]: ")

    if(params.fused) {
      Engine.dnnComputing.invokeAndWait2((0 until 1).map(i => () => {
        if (params.infer) {
          model.evaluate()
          model.asInstanceOf[DnnGraph].compile(InferencePhase)
        }
        else {
          model.asInstanceOf[DnnGraph].compile(TrainingPhase)
        }

        for (i <- 1 to 30) {
          model.forward(miniBatch.getInput())
          if (!params.infer) {
            model.backward(miniBatch.getInput(), miniBatch.getTarget())
          }
        }

        val start = System.nanoTime()
        for (i <- 1 to params.iteration) {
          if (params.verbose) {
            // val s1 = System.nanoTime()
            model.forward(miniBatch.getInput())
            if (!params.infer) {
              model.backward(miniBatch.getInput(), miniBatch.getTarget())
            }
            // val e1 = System.nanoTime()
            getTopTimes(model.getTimes())
            // println(s"iteration time ${(e1 - s1)/1e9}")
            println('\n')
            model.resetTimes()
          }
          else {
            model.forward(miniBatch.getInput())
            if (!params.infer) {
              model.backward(miniBatch.getInput(), miniBatch.getTarget())
            }
          }
        }
        val end = System.nanoTime()

        println(s"Avg iteration time: ${(end - start)/(params.iteration * 1e9)} sec(s)")

        logger.info(s"[PTBModel-LSTM MKLDNN (fused)] result: ")
        logger.info(s"Average Throughput" +
          s" is ${params.batchSize.toDouble * params.iteration / (end - start) * 1e9}" +
          s" records / second.")
      }))
    }

    else {
      if (params.infer) {
        model.evaluate()
      }

      for (i <- 1 to 30) {
        model.forward(miniBatch.getInput())
        if (!params.infer) {
          model.backward(miniBatch.getInput(), miniBatch.getTarget())
        }
      }

      val start = System.nanoTime()
      for (i <- 1 to params.iteration) {
        if (params.verbose) {
          // val s1 = System.nanoTime()
          model.forward(miniBatch.getInput())
          if (!params.infer) {
            model.backward(miniBatch.getInput(), miniBatch.getTarget())
          }
          // val e1 = System.nanoTime()
          getTopTimes(model.getTimes())
          // println(s"iteration time ${(e1 - s1)/1e9}")
          println('\n')
          model.resetTimes()
        }
        else {
          model.forward(miniBatch.getInput())
          if (!params.infer) {
            model.backward(miniBatch.getInput(), miniBatch.getTarget())
          }
        }
      }
      val end = System.nanoTime()

      println(s"Avg iteration time: ${(end - start)/(params.iteration * 1e9)} sec(s)")

      logger.info(s"[PTBModel-LSTM MKLDNN (NOT fused)] result: ")
      logger.info(s"Average Throughput" +
        s" is ${params.batchSize.toDouble * params.iteration / (end - start) * 1e9}" +
        s" records / second.")
    }
  }

  def main(argv: Array[String]): Unit = {
    parser.parse(argv, new DistriPerfParams()).foreach { params =>
      println("batchSize = " + params.batchSize)
      println("iterations = " + params.iteration)
      println("seqLength = " + params.seqLength)
      println("inputSize = " + params.vocabSize)
      println("hiddenSize = " + params.hiddenSize)
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

      val input = Tensor[Float](Array(params.batchSize, params.seqLength))
      for (n <- 1 to params.batchSize) {
        for (t <- 1 to params.seqLength) {
          input(Array(n, t)) = scala.util.Random.nextInt(params.vocabSize) + 1
        }
      }

      RNG.setSeed(100)
      val gradOutput = Tensor[Float](Array(params.batchSize, params.seqLength, params.vocabSize))
        .rand()

      val minibatch = MiniBatch(input, gradOutput)

      if (params.engineType == "mkldnn") {
        dnnPredict(params, sc, minibatch)
      }

      if (params.engineType == "blas") {
        blasPredict(params, sc, minibatch)
      }
    }
  }
}

case class DistriPerfParams (
  batchSize: Int = 20,
  iteration: Int = 100,
  seqLength: Int = 300,
  vocabSize: Int = 800,
  hiddenSize: Int = 800,
  numLayers: Int = 1,
  threadNum: Int = 1,
  fused: Boolean = false,
  verbose: Boolean = false,
  infer: Boolean = false,
  engineType: String = "mkldnn",
  model: String = "vanilla_lstm",
  threadPredict: Boolean = true
)

