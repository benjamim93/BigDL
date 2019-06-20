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
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.mkl.{AlgKind, Direction, Memory}
import com.intel.analytics.bigdl.models.inception.Inception_v1_NoAuxClassifier
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn
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

object DistriPerf_inf {
  val logger = Logger.getLogger(getClass)

  val parser = new OptionParser[DistriPerfParams]("BigDL w/ Dnn Local Model Performance Test") {
    opt[String]('e', "engine")
      .text("lstm_nn | lstm_mkldnn")
      .action((v, p) => p.copy(engineType = v))
    /* opt[String]('p', "path")
      .text("model you want, vgg16 | resnet50 | vgg16_graph | resnet50_graph")
      .action((v, p) => p.copy(modelPath = v)) */
    opt[Int]('b', "batchSize")
      .text("Batch size of input data")
      .action((v, p) => p.copy(batchSize = v))
    opt[Int]('i', "iteration")
      .text("Iteration of perf test. The result will be average of each iteration time cost")
      .action((v, p) => p.copy(iteration = v))
    opt[Boolean]('t', "threadPredict")
      .text("Whether thread predict")
      .action((v, p) => p.copy(threadPredict = v))
    opt[Int]('c', "commonSize")
      .text("The common size of input and output")
      .action((v, p) => p.copy(commonSize = v))
    opt[Int]('s', "seqLength")
      .text("The length of sequence")
      .action((v, p) => p.copy(seqLength = v))
    opt[String]('x', "threadNum")
      .text("Number of threads")
      .action((v, p) => p.copy(threadNum = v))
    opt[String]('m', "blasModelType")
      .text("Type of blas model")
      .action((v, p) => p.copy(blasModelType = v))
  }

  def getTopTimes(times: Array[(AbstractModule[_ <: Activity, _ <: Activity, Float],
    Long, Long)], allSum: Long): Unit = {
    var forwardSum = 0L
    var backwardSum = 0L
    times.foreach(x => {
      forwardSum += x._2
      backwardSum += x._3
    })
    println(s"forwardSum = ${forwardSum/1e9}" +
      s" realTime ${allSum/1e9} backwardSum = ${backwardSum/1e9}")

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

  def predict(model: Module[Float], input: MiniBatch[Float],
              params: DistriPerfParams): Unit = {
    val subModelNumber = Engine.getEngineType() match {
      case MklBlas => Engine.coreNumber()
      case MklDnn => 1
    }
    model.evaluate()
    val workingModels = if (subModelNumber != 1) {
      val wb = Util.getAndClearWeightBias(model.parameters())
      val models = (1 to subModelNumber).map(i => {
        logger.info(s"Clone $i model...")
        val m = model.cloneModule()
        Util.putWeightBias(wb, m)
        m.evaluate()
        m
      }).toArray
      Util.putWeightBias(wb, model)
      models
    } else {
      Array(model)
    }

    val inputSize = input.size()
    val stackSize = input.size() / subModelNumber
    val extraSize = input.size() % subModelNumber
    val parallelism = if (stackSize == 0) extraSize else subModelNumber
    val inputBuffer = new Array[MiniBatch[Float]](parallelism)
    var b = 0
    while (b < parallelism) {
      val offset = b * stackSize + math.min(b, extraSize) + 1
      val length = stackSize + (if (b < extraSize) 1 else 0)
      inputBuffer(b) = input.slice(offset, length)
      b += 1
    }

    // warm up
    println(s"engine default pool size ${Engine.default.getPoolSize}")
    val warmup = 20
    val warmpResults = Engine.default.invoke((0 until subModelNumber).map(i =>
      () => {
        val localModel = workingModels(i).evaluate()
        val data = inputBuffer(i)
        for (i <- 0 to warmup) {
          val output = localModel.forward(data.getInput())
        }
        1
      }))
    Engine.default.sync(warmpResults)
    println("start predict throughput test")
    val start = System.nanoTime()
    for (i <- 0 to params.iteration) {
      val results = Engine.default.invoke((0 until subModelNumber).map(i =>
        () => {
          val localModel = workingModels(i).evaluate()
          val data = inputBuffer(i)
          val s1 = System.nanoTime()
          val output = localModel.forward(data.getInput())
          //          val e1 = System.nanoTime() - s1
          //          getTopTimes(localModel.getTimes(), e1)
          //          println(s"iteration time ${e1/1e9}")
          //          localModel.resetTimes()
          1
        }))
      Engine.default.sync(results)
    }

    val end = System.nanoTime()

    logger.info(s"Use java thread ${params.model} isNNRecurrent" +
      s" ${model.isInstanceOf[nn.Recurrent[Float]]} " +
      s"isMKLDNNLSTM ${model.isInstanceOf[RNN]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} " + s"Average Throughput" +
      s"is ${params.batchSize.toDouble * params.iteration / (end - start) * 1e9} record / second."
    )
  }

  def dnnPredict(model: Module[Float], input: MiniBatch[Float],
                 params: DistriPerfParams): Unit = {
    println("\nstart predict throughput test [Uni L2R 1 Layer]: ")
    var time : Long = 0

    val f = AlgKind.EltwiseTanh
    var direction = Direction.UnidirectionalLeft2Right
    val inputFormat = HeapData(Array(params.seqLength, params.batchSize,
      params.commonSize), Memory.Format.tnc)

    val lstm1 = RNN(
      AlgKind.VanillaLstm, params.commonSize, params.commonSize, f, direction, layers = 1)

    // Engine.dnnComputing.invokeAndWait2(Array(0).map(_ => () => {
    lstm1.evaluate()
    lstm1.setRuntime(new MklDnnRuntime)
    lstm1.initFwdPrimitives(Array(inputFormat), InferencePhase)

    for (i <- 1 to params.iteration) {
      // println("iteration: " + i)
      val start = System.nanoTime()
      lstm1.evaluate()
      val output = lstm1.forward(input.getInput())
      val end = System.nanoTime()
      time += (end - start)
    }
    // }))

    logger.info("[Uni L2R 1 Layer] result: ")
    logger.info(s"Use java thread ${params.model} isNNRecurrent" +
      s" ${model.isInstanceOf[nn.Recurrent[Float]]} " +
      s"isMKLDNNLSTM ${model.isInstanceOf[RNN]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} " + s"Average Throughput" +
      s"is ${params.batchSize.toDouble * params.iteration / (time) * 1e9} record / second."
    )
    println("========================================================================\n\n")

    time = 0

    direction = Direction.BidirectionalConcat
    val lstm2 = RNN(
      AlgKind.VanillaLstm, params.commonSize, params.commonSize, f, direction, layers = 1)

    lstm2.evaluate()
    lstm2.setRuntime(new MklDnnRuntime)
    lstm2.initFwdPrimitives(Array(inputFormat), InferencePhase)

    for (i <- 1 to params.iteration) {
      val start = System.nanoTime()
      lstm2.evaluate()
      val output = lstm2.forward(input.getInput())
      val end = System.nanoTime()
      time += (end - start)
    }

    logger.info("[Bi Concat 1 Layer] result: ")
    logger.info(s"Use java thread ${params.model} isNNRecurrent" +
      s" ${model.isInstanceOf[nn.Recurrent[Float]]} " +
      s"isMKLDNNLSTM ${model.isInstanceOf[RNN]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} " + s"Average Throughput" +
      s"is ${params.batchSize.toDouble * params.iteration / (time) * 1e9} record / second."
    )
    println("========================================================================\n\n")

    time = 0

    direction = Direction.BidirectionalSum
    val lstm3 = RNN(
      AlgKind.VanillaLstm, params.commonSize, params.commonSize, f, direction, layers = 1)

    lstm3.evaluate()
    lstm3.setRuntime(new MklDnnRuntime)
    lstm3.initFwdPrimitives(Array(inputFormat), InferencePhase)

    for (i <- 1 to params.iteration) {
      // println("iteration: " + i)
      val start = System.nanoTime()
      lstm3.evaluate()
      val output = lstm3.forward(input.getInput())
      val end = System.nanoTime()
      time += (end - start)
    }

    logger.info("[Bi Sum 1 Layer] result: ")
    logger.info(s"Use java thread ${params.model} isNNRecurrent" +
      s" ${model.isInstanceOf[nn.Recurrent[Float]]} " +
      s"isMKLDNNLSTM ${model.isInstanceOf[RNN]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} " + s"Average Throughput" +
      s"is ${params.batchSize.toDouble * params.iteration / (time) * 1e9} record / second."
    )
    println("========================================================================\n\n")

    time = 0

    direction = Direction.UnidirectionalLeft2Right
    val lstm4 = RNN(
      AlgKind.VanillaLstm, params.commonSize, params.commonSize, f, direction, layers = 5)

    lstm4.evaluate()
    lstm4.setRuntime(new MklDnnRuntime)
    lstm4.initFwdPrimitives(Array(inputFormat), InferencePhase)

    for (i <- 1 to params.iteration) {
      val start = System.nanoTime()
      lstm4.evaluate()
      val output = lstm4.forward(input.getInput())
      val end = System.nanoTime()
      time += (end - start)
    }

    logger.info("[Uni L2R 5 Layers] result: ")
    logger.info(s"Use java thread ${params.model} isNNRecurrent" +
      s" ${model.isInstanceOf[nn.Recurrent[Float]]} " +
      s"isMKLDNNLSTM ${model.isInstanceOf[RNN]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} " + s"Average Throughput" +
      s"is ${params.batchSize.toDouble * params.iteration / (time) * 1e9} record / second."
    )
    println("========================================================================\n\n")

    /*
    time = 0

    direction = Direction.BidirectionalConcat
    val lstm5 = RNN(
      AlgKind.VanillaLstm, params.commonSize, params.commonSize, f, direction, layers = 5)

    lstm5.setRuntime(new MklDnnRuntime)
    lstm5.initFwdPrimitives(Array(inputFormat), InferencePhase)

    for (i <- 1 to params.iteration) {
      val start = System.nanoTime()
      lstm5.evaluate()
      val output = lstm5.forward(input.getInput())
      val end = System.nanoTime()
      time += (end - start)
    }

    logger.info("[Bi Concat 5 Layers] result: ")
    logger.info(s"Use java thread ${params.model} isNNRecurrent" +
      s" ${model.isInstanceOf[nn.Recurrent[Float]]} " +
      s"isMKLDNNLSTM ${model.isInstanceOf[RNN]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} " + s"Average Throughput" +
      s"is ${params.batchSize.toDouble * params.iteration / (time) * 1e9} record / second."
    )
    println("========================================================================\n\n")

    time = 0

    direction = Direction.BidirectionalSum
    val lstm6 = RNN(
      AlgKind.VanillaLstm, params.commonSize, params.commonSize, f, direction, layers = 5)

    lstm6.setRuntime(new MklDnnRuntime)
    lstm6.initFwdPrimitives(Array(inputFormat), InferencePhase)

    for (i <- 1 to params.iteration) {
      val start = System.nanoTime()
      lstm6.evaluate()
      val output = lstm6.forward(input.getInput())
      val end = System.nanoTime()
      time += (end - start)
    }

    logger.info("[Bi Sum 5 Layers] result: ")
    logger.info(s"Use java thread ${params.model} isNNRecurrent" +
      s" ${model.isInstanceOf[nn.Recurrent[Float]]} " +
      s"isMKLDNNLSTM ${model.isInstanceOf[RNN]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} " + s"Average Throughput" +
      s"is ${params.batchSize.toDouble * params.iteration / (time) * 1e9} record / second."
    )
    println("========================================================================\n\n")
    */
  }

  def threadPredict(params: DistriPerfParams, sc: SparkContext, modelLoad: Module[Float]): Unit = {
    println("inputSize = " + params.commonSize)
    println("hiddenSize = " + params.commonSize)
    println("batchSize = " + params.batchSize)
    println("seqLength = " + params.seqLength)
    println("iterations = " + params.iteration)
    println("engineType = " + params.engineType)

    //    if(params.engineType == "lstm_nn") {
    //      require(modelLoad.isInstanceOf[nn.Recurrent[Float]])
    //    }

    if(params.engineType == "lstm_mkldnn") {
      require(modelLoad.isInstanceOf[RNN])
    }

    var inputShape: Array[Int] = null
    var outputShape: Array[Int] = null

    params.engineType match {
      case "lstm_nn" =>
        inputShape = Array(params.batchSize, params.seqLength, params.commonSize)
        outputShape = Array(params.batchSize, params.seqLength, params.commonSize)
      case "lstm_mkldnn" =>
        inputShape = Array(params.seqLength, params.batchSize, params.commonSize)
        outputShape = Array(params.seqLength, params.batchSize, params.commonSize)
      case _ => throw new UnsupportedOperationException(s"Unknown model ${params.engineType}")
    }

    val miniBatch = MiniBatch(Tensor(inputShape).rand(), Tensor(outputShape).rand())
    if (Engine.getEngineType() == MklDnn) {
      dnnPredict(modelLoad, miniBatch, params)
    } else {
      predict(modelLoad, miniBatch, params)
    }
  }

  def main(argv: Array[String]): Unit = {
    parser.parse(argv, new DistriPerfParams()).foreach { params =>
      if (params.engineType == "lstm_mkldnn") {
        System.setProperty("bigdl.engineType", "mkldnn")
        System.setProperty("bigdl.mklNumThreads", params.threadNum)
      }

      if (params.engineType == "lstm_nn") {
        System.setProperty("bigdl.engineType", "mklblas")
      }

      val conf = Engine.createSparkConf()
        .setAppName("Test perf")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)

      Engine.init
      Engine.setNodeAndCore(Engine.nodeNumber(), Engine.coreNumber())

      // NN LSTM
      if(params.engineType == "lstm_nn") {
        println("blas")
        Engine.setEngineType(MklBlas)

        if(params.blasModelType == "unil2r1") {
          val nn_model = nn.Recurrent[Float]().add(nn.LSTM(params.commonSize, params.commonSize))
          threadPredict(params, sc, nn_model)
          println("[Uni L2R 1 layer] result\n\n ")
        }

        if(params.blasModelType == "biconcat1") {
          val nn_model2 = nn.BiRecurrent[Float](nn.JoinTable[Float](3, 0)
            .asInstanceOf[AbstractModule[Table, Tensor[Float], Float]])
            .add(nn.LSTM(params.commonSize, params.commonSize))

          threadPredict(params, sc, nn_model2)
          println("[Bi Concat 1 layer] result\n\n ")
        }

        if(params.blasModelType == "bisum1") {
          val nn_model3 = nn.BiRecurrent[Float](nn.CAddTable()
            .asInstanceOf[AbstractModule[Table, Tensor[Float], Float]])
            .add(nn.LSTM(params.commonSize, params.commonSize))

          threadPredict(params, sc, nn_model3)
          println("[Bi Sum 1 layer] result\n\n ")
        }


        if(params.blasModelType == "unil2r5") {
          val nn_input = nn.Input()
          var nn_lstm = nn.Recurrent[Float]()
            .add(nn.LSTM(params.commonSize, params.commonSize)).inputs(nn_input)

          for (i <- 1 until 5) {
            nn_lstm = nn.Recurrent[Float]()
              .add(nn.LSTM(params.commonSize, params.commonSize)).inputs(nn_lstm)
          }

          val nn_model4 = nn.Graph(nn_input, nn_lstm)
          threadPredict(params, sc, nn_model4)
          println("[Uni L2R 5 layers] result\n\n ")
        }
      }

      // MKLDNN LSTM
      if(params.engineType == "lstm_mkldnn") {
        println("\nmkldnn")
        println("Thread number: " + params.threadNum)

        val f = AlgKind.EltwiseTanh
        var direction = Direction.UnidirectionalLeft2Right

        val inputFormat = HeapData(Array(params.seqLength, params.batchSize,
          params.commonSize), Memory.Format.tnc)

        val lstm1 = RNN(
          AlgKind.VanillaLstm, params.commonSize, params.commonSize, f, direction, layers = 1)

        val modelLoad = lstm1
        threadPredict(params, sc, modelLoad)
      }
    }
  }
}

case class DistriPerfInfParams (
  batchSize: Int = 20,
  iteration: Int = 100,
  commonSize: Int = 800,
  seqLength: Int = 300,
  threadNum: String = "1",
  engineType: String = "lstm_mkldnn",
  model: String = "vanilla_lstm",
  blasModelType: String = "unil2r",
  threadPredict: Boolean = true
)


