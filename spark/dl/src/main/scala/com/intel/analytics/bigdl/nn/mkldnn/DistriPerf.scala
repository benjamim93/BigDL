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
  /*
  val seqLength = 300
  val inputSize = 800
  val hiddenSize = 800
  */

  val common_n_layers = 1
  val lstm_n_gates = 4

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
      // println("iteration: " + i)
    }

    val end = System.nanoTime()

    logger.info(s"Use java thread ${params.model} isNNRecurrent" +
      s" ${model.isInstanceOf[nn.Recurrent[Float]]} " +
      s"isMKLDNNLSTM ${model.isInstanceOf[LSTM]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} " + s"Average Throughput" +
      s"is ${params.batchSize.toDouble * params.iteration / (end - start) * 1e9} record / second."
    )
  }

  def dnnPredict(model: Module[Float], input: MiniBatch[Float],
              params: DistriPerfParams): Unit = {
    println("\nstart predict throughput test: ")
    var time : Long = 0

    val f = AlgKind.EltwiseTanh
    var direction = Direction.UnidirectionalLeft2Right
    val inputFormat = HeapData(Array(params.seqLength, params.batchSize,
      params.commonSize), Memory.Format.tnc)
    val lstm1 = LSTM(params.commonSize, params.commonSize, f, direction, layers = 1)

    Engine.dnnComputing.invokeAndWait2(Array(0).map(_ => () => {
      lstm1.setRuntime(new MklDnnRuntime)
      lstm1.initFwdPrimitives(Array(inputFormat), InferencePhase)

      for (i <- 1 to params.iteration) {
        println("iteration: " + i)
        val start = System.nanoTime()
        lstm1.evaluate()
        val output = lstm1.forward(input.getInput())
        val end = System.nanoTime()
        time += (end - start)
//        println("forward() time consumption: " + (end -start) + "\n\n")
//        println("s/f portion: " + (end -start) + "\n\n")
      }
    }))

    logger.info(s"Use java thread ${params.model} isNNRecurrent" +
      s" ${model.isInstanceOf[nn.Recurrent[Float]]} " +
      s"isMKLDNNLSTM ${model.isInstanceOf[LSTM]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} " + s"Average Throughput" +
      s"is ${params.batchSize.toDouble * params.iteration / (time) * 1e9} record / second."
    )
  }

  def threadPredict(params: DistriPerfParams, sc: SparkContext, modelLoad: Module[Float]): Unit = {
    println("inputSize = " + params.commonSize)
    println("hiddenSize = " + params.commonSize)
    println("batchSize = " + params.batchSize)
    println("seqLength = " + params.seqLength)
    println("iterations = " + params.iteration)
    println("engineType = " + params.engineType)

    if(params.engineType == "lstm_nn") {
      require(modelLoad.isInstanceOf[nn.Recurrent[Float]])
    }

    if(params.engineType == "lstm_mkldnn") {
      require(modelLoad.isInstanceOf[LSTM])
    }

    // println(modelLoad.isInstanceOf[nn.Recurrent[Float]])
    // println(modelLoad.isInstanceOf[LSTM])

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

      /*
      var initWeight = Tensor[Float](
        Array(common_n_layers, 1,
          params.commonSize, lstm_n_gates, params.commonSize)).rand(-1.0, 1.0)
      var initWeightIter = Tensor[Float](
        Array(common_n_layers, 1,
          params.commonSize, lstm_n_gates, params.commonSize)).rand(-1.0, 1.0)
      var initBias = Tensor[Float](
        Array(common_n_layers, 1,
          lstm_n_gates, params.commonSize)).rand(-1.0, 1.0)
          */

      // NN LSTM
      if(params.engineType == "lstm_nn") {
        println("blas")
        Engine.setEngineType(MklBlas)

        /*
        initWeight = initWeight.resize(Array(inputSize, lstm_n_gates, hiddenSize))
          .transpose(1, 2).transpose(2, 3)
        initWeightIter = initWeightIter.resize(Array(hiddenSize, lstm_n_gates, hiddenSize))
          .transpose(1, 2).transpose(2, 3)
        initBias = initBias.resize(Array(lstm_n_gates, hiddenSize))
        */

        /**
          * MKLDNN Gate 1 -> nn/LSTM Gate 1
          * MKLDNN Gate 2 -> nn/LSTM Gate 3
          * MKLDNN Gate 3 -> nn/LSTM Gate 2
          * MKLDNN Gate 4 -> nn/LSTM Gate 4
          *
          * uniParams(0) -> input weights
          * uniParams(1) -> bias
          * uniParams(2) -> hidden weights
          */

        /*
        val concat = nn.JoinTable(1, 4)
        initWeight = concat.forward(T(initWeight(1), initWeight(3),
          initWeight(2), initWeight(4))).asInstanceOf[Tensor[Float]].clone()
        initWeightIter = concat.forward(T(initWeightIter(1), initWeightIter(3),
          initWeightIter(2), initWeightIter(4))).asInstanceOf[Tensor[Float]].clone()
        initBias = concat.forward(T(initBias(1), initBias(3), initBias(2), initBias(4)))
          .asInstanceOf[Tensor[Float]].clone()
         */

        val nn_model = nn.Recurrent().add(nn.LSTM(params.commonSize, params.commonSize))

        /*
        val uniParams = nn_model.parameters()._1
        initWeight = initWeight.resizeAs(uniParams(0))
        initBias = initBias.resizeAs(uniParams(1))
        initWeightIter = initWeightIter.resizeAs(uniParams(2))

        uniParams(0).copy(initWeight)
        uniParams(1).copy(initBias)
        uniParams(2).copy(initWeightIter)
        */

        val modelLoad = nn_model
        threadPredict(params, sc, modelLoad)
      }

      // MKLDNN LSTM
      if(params.engineType == "lstm_mkldnn") {
        println("\nmkldnn")
        println("Thread number: " + params.threadNum)

        val f = AlgKind.EltwiseTanh
        var direction = Direction.UnidirectionalLeft2Right

        val inputFormat = HeapData(Array(params.seqLength, params.batchSize,
          params.commonSize), Memory.Format.tnc)

        /*
        val lstm1 = LSTM(params.commonSize, params.commonSize, f, direction,
          initWeight = initWeight, initWeightIter = initWeightIter, initBias = initBias)
          */

        val lstm1 = LSTM(params.commonSize, params.commonSize, f, direction, layers = 1)

        val modelLoad = lstm1
        threadPredict(params, sc, modelLoad)
      }
    }
  }
}

case class DistriPerfParams (
    batchSize: Int = 20,
    iteration: Int = 100,
    commonSize: Int = 800,
    seqLength: Int = 300,
    threadNum: String = "1",
    engineType: String = "lstm_mkldnn",
    model: String = "vanilla_lstm",
    threadPredict: Boolean = true
  )


