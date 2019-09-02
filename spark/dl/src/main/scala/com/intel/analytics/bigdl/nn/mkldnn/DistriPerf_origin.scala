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
import com.intel.analytics.bigdl.mkl.hardware.Affinity
import com.intel.analytics.bigdl.mkl.{AlgKind, Direction, MKL, Memory}
import com.intel.analytics.bigdl.models.inception.Inception_v1_NoAuxClassifier
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.mkldnn.DistriPerf.logger
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

object DistriPerfOrigin {
  val logger = Logger.getLogger(getClass)

  val parser = new OptionParser[DistriPerfOriginParams](
    "BigDL w/ Dnn Local Model Performance Test") {
    opt[String]('e', "engine")
      .text("rnn_nn | rnn_mkldnn")
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
              params: DistriPerfOriginParams): Unit = {
    val subModelNumber = Engine.getEngineType() match {
      case MklBlas => Engine.coreNumber()
      case MklDnn => 1
    }
    // model.evaluate()
    val workingModels = if (subModelNumber != 1) {
      val wb = Util.getAndClearWeightBias(model.parameters())
      val models = (1 to subModelNumber).map(i => {
        logger.info(s"Clone $i model...")
        val m = model.cloneModule()
        Util.putWeightBias(wb, m)
        // m.evaluate()
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

    val gradOutputShape =
      if (params.blasModelType == "unil2r1"
        || params.blasModelType == "unil2r5"
        || params.blasModelType == "bisum1") {
        Array(params.batchSize, params.seqLength, params.commonSize)
      }
      else {
        Array( params.batchSize, params.seqLength, 2 * params.commonSize)
      }

    // warm up
    println(s"engine default pool size ${Engine.default.getPoolSize}")
    val warmup = 20
    val warmpResults = Engine.default.invoke((0 until subModelNumber).map(i =>
      () => {
        // val localModel = workingModels(i).evaluate()
        val localModel = workingModels(i)
        val data = inputBuffer(i)
        val datainput = data.getInput()

        val gradOutput = Tensor(
          Array(datainput.toTensor.size(1), gradOutputShape(1), gradOutputShape(2)))
          .rand(1.0, 1.0)

        for (i <- 0 to warmup) {
          val output = localModel.forward(datainput)
          val gradinput = localModel.backward(datainput, gradOutput)
        }
        1
      }))
    Engine.default.sync(warmpResults)
    println("start predict throughput test")

    val start = System.nanoTime()
    for (i <- 0 to params.iteration) {
      val results = Engine.default.invoke((0 until subModelNumber).map(i =>
        () => {
          // val localModel = workingModels(i).evaluate()
          val localModel = workingModels(i)
          val data = inputBuffer(i)
          val datainput = data.getInput()

          val gradOutput = Tensor(
            Array(datainput.toTensor.size(1), gradOutputShape(1), gradOutputShape(2)))
            .rand(1.0, 1.0)

          val output = localModel.forward(datainput)
          val gradinput = localModel.backward(datainput, gradOutput)

          // time += (end - start)

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
      s"isMKLDNNGRU ${model.isInstanceOf[RNN]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} " + s"Average Throughput" +
      s"is ${params.batchSize.toDouble * params.iteration / (end - start) * 1e9} " +
      s"record / second."
    )
  }

  def dnnPredict(model: Module[Float], input: MiniBatch[Float],
                 params: DistriPerfOriginParams): Unit = {
    println("\nstart predict throughput test [Uni L2R 1 Layer]: ")
    val f = AlgKind.EltwiseTanh
    var direction = Direction.UnidirectionalLeft2Right

    val inputFormat = HeapData(Array(params.seqLength, params.batchSize,
      params.commonSize), Memory.Format.tnc)
    val input_t = input.getInput()

    var gradOutputFormat = HeapData(Array(params.seqLength, params.batchSize,
      params.commonSize), Memory.Format.tnc)
    var gradOutput_t = Tensor(gradOutputFormat.shape).rand(1.0, 1.0)

    println("========================================================================\n\n")

    val rnn1 = RNN(AlgKind.VanillaGru, params.commonSize, params.commonSize,
      f, direction, layers = 1)

    val start1 = System.nanoTime()
    Engine.dnnComputing.invokeAndWait2((0 until 1).map(i => () => {
      rnn1.setRuntime(new MklDnnRuntime)
      rnn1.initFwdPrimitives(Array(inputFormat), TrainingPhase)
      rnn1.initBwdPrimitives(Array(gradOutputFormat), TrainingPhase)
      for (i <- 1 to params.iteration) {
        val output = rnn1.forward(input_t)
        val gradInput = rnn1.backward(input_t, gradOutput_t)
      }
    }))
    val end1 = System.nanoTime()

    logger.info("[Uni L2R 1 Layer FWD + BWD] result: ")
    logger.info(s"Use java thread ${params.model} isNNRecurrent " +
      s"${model.isInstanceOf[nn.Recurrent[Float]]} " +
      s"isMKLDNNGRU ${model.isInstanceOf[RNN]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} ")
    logger.info(s"Average Throughput is: " +
      s"${params.batchSize.toDouble * params.iteration / (end1 - start1) * 1e9} " +
      s"record / second.")

    println("========================================================================\n\n")

    gradOutputFormat = HeapData(Array(params.seqLength, params.batchSize,
      2 * params.commonSize), Memory.Format.tnc)
    gradOutput_t = Tensor(gradOutputFormat.shape).rand(1.0, 1.0)

    direction = Direction.BidirectionalConcat
    val gru2 = RNN(AlgKind.VanillaGru, params.commonSize, params.commonSize,
      f, direction, layers = 1)

    val start2 = System.nanoTime()
    Engine.dnnComputing.invokeAndWait2((0 until 1).map(i => () => {
      gru2.setRuntime(new MklDnnRuntime)
      gru2.initFwdPrimitives(Array(inputFormat), TrainingPhase)
      gru2.initBwdPrimitives(Array(gradOutputFormat), TrainingPhase)
      for (i <- 1 to params.iteration) {
        val output = gru2.forward(input_t)
        val gradInput = gru2.backward(input_t, gradOutput_t)
      }
    }))
    val end2 = System.nanoTime()

    logger.info("[Bi Concat 1 Layer FWD + BWD] result: ")
    logger.info(s"Use java thread ${params.model} isNNRecurrent " +
      s"${model.isInstanceOf[nn.Recurrent[Float]]} " +
      s"isMKLDNNGRU ${model.isInstanceOf[RNN]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} ")
    logger.info(s"Average Throughput is: " +
      s"${params.batchSize.toDouble * params.iteration / (end2 - start2) * 1e9} " +
      s"record / second.")

    println("========================================================================\n\n")

    gradOutputFormat = HeapData(Array(params.seqLength, params.batchSize,
      2 * params.commonSize), Memory.Format.tnc)
    gradOutput_t = Tensor(gradOutputFormat.shape).rand(1.0, 1.0)

    direction = Direction.BidirectionalSum
    val gru3 = RNN(AlgKind.VanillaGru, params.commonSize, params.commonSize,
      f, direction, layers = 1)

    val start3 = System.nanoTime()
    Engine.dnnComputing.invokeAndWait2((0 until 1).map(i => () => {
      gru3.setRuntime(new MklDnnRuntime)
      gru3.initFwdPrimitives(Array(inputFormat), TrainingPhase)
      gru3.initBwdPrimitives(Array(gradOutputFormat), TrainingPhase)
      for (i <- 1 to params.iteration) {
        val output = gru3.forward(input_t)
        val gradInput = gru3.backward(input_t, gradOutput_t)
      }
    }))
    val end3 = System.nanoTime()

    logger.info("[Bi Sum 1 Layer FWD + BWD] result: ")
    logger.info(s"Use java thread ${params.model} isNNRecurrent " +
      s"${model.isInstanceOf[nn.Recurrent[Float]]} " +
      s"isMKLDNNGRU ${model.isInstanceOf[RNN]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} ")
    logger.info(s"Average Throughput is: " +
      s"${params.batchSize.toDouble * params.iteration / (end3 - start3) * 1e9} " +
      s"record / second.")

    println("========================================================================\n\n")

    gradOutputFormat = HeapData(Array(params.seqLength, params.batchSize,
      params.commonSize), Memory.Format.tnc)
    gradOutput_t = Tensor(gradOutputFormat.shape).rand(1.0, 1.0)

    direction = Direction.UnidirectionalLeft2Right
    val gru4 = RNN(AlgKind.VanillaGru, params.commonSize, params.commonSize,
      f, direction, layers = 5)

    val start4 = System.nanoTime()
    Engine.dnnComputing.invokeAndWait2((0 until 1).map(i => () => {
      gru4.setRuntime(new MklDnnRuntime)
      gru4.initFwdPrimitives(Array(inputFormat), TrainingPhase)
      gru4.initBwdPrimitives(Array(gradOutputFormat), TrainingPhase)
      for (i <- 1 to params.iteration) {
        val output = gru4.forward(input_t)
        val gradInput = gru4.backward(input_t, gradOutput_t)
      }
    }))
    val end4 = System.nanoTime()

    logger.info("[Uni L2R 5 Layers FWD + BWD] result: ")
    logger.info(s"Use java thread ${params.model} isNNRecurrent" +
      s"${model.isInstanceOf[nn.Recurrent[Float]]} " +
      s"isMKLDNNGRU ${model.isInstanceOf[RNN]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} ")
    logger.info(s"Average Throughput is: " +
      s"${params.batchSize.toDouble * params.iteration / (end4 - start4) * 1e9}" +
      s"record / second.")

    println("========================================================================\n\n")

    val gru1_inf = RNN(AlgKind.VanillaGru, params.commonSize, params.commonSize,
      f, direction, layers = 1)

    val start1_inf = System.nanoTime()
    Engine.dnnComputing.invokeAndWait2((0 until 1).map(i => () => {
      gru1_inf.evaluate()
      gru1_inf.setRuntime(new MklDnnRuntime)
      gru1_inf.initFwdPrimitives(Array(inputFormat), InferencePhase)
      for (i <- 1 to params.iteration) {
        val output = gru1_inf.forward(input_t)
      }
    }))
    val end1_inf = System.nanoTime()

    logger.info("[Uni L2R 1 Layer FWD] result: ")
    logger.info(s"Use java thread ${params.model} isNNRecurrent " +
      s"${model.isInstanceOf[nn.Recurrent[Float]]} " +
      s"isMKLDNNGRU ${model.isInstanceOf[RNN]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} ")
    logger.info(s"Average Throughput is: " +
      s"${params.batchSize.toDouble * params.iteration / (end1_inf - start1_inf) * 1e9} " +
      s"record / second.")

    println("========================================================================\n\n")

    gradOutputFormat = HeapData(Array(params.seqLength, params.batchSize,
      2 * params.commonSize), Memory.Format.tnc)
    gradOutput_t = Tensor(gradOutputFormat.shape).rand(1.0, 1.0)

    direction = Direction.BidirectionalConcat
    val gru2_inf = RNN(AlgKind.VanillaGru, params.commonSize, params.commonSize,
      f, direction, layers = 1)

    val start2_inf = System.nanoTime()
    Engine.dnnComputing.invokeAndWait2((0 until 1).map(i => () => {
      gru2_inf.evaluate()
      gru2_inf.setRuntime(new MklDnnRuntime)
      gru2_inf.initFwdPrimitives(Array(inputFormat), InferencePhase)
      for (i <- 1 to params.iteration) {
        val output = gru2_inf.forward(input_t)
      }
    }))
    val end2_inf = System.nanoTime()

    logger.info("[Bi Concat 1 Layer FWD] result: ")
    logger.info(s"Use java thread ${params.model} isNNRecurrent " +
      s"${model.isInstanceOf[nn.Recurrent[Float]]} " +
      s"isMKLDNNGRU ${model.isInstanceOf[RNN]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} ")
    logger.info(s"Average Throughput is: " +
      s"${params.batchSize.toDouble * params.iteration / (end2_inf - start2_inf) * 1e9} " +
      s"record / second.")

    println("========================================================================\n\n")

    gradOutputFormat = HeapData(Array(params.seqLength, params.batchSize,
      2 * params.commonSize), Memory.Format.tnc)
    gradOutput_t = Tensor(gradOutputFormat.shape).rand(1.0, 1.0)

    direction = Direction.BidirectionalSum
    val gru3_inf = RNN(AlgKind.VanillaGru, params.commonSize, params.commonSize,
      f, direction, layers = 1)

    val start3_inf = System.nanoTime()
    Engine.dnnComputing.invokeAndWait2((0 until 1).map(i => () => {
      gru3_inf.evaluate()
      gru3_inf.setRuntime(new MklDnnRuntime)
      gru3_inf.initFwdPrimitives(Array(inputFormat), InferencePhase)
      for (i <- 1 to params.iteration) {
        val output = gru3_inf.forward(input_t)
      }
    }))
    val end3_inf = System.nanoTime()

    logger.info("[Bi Sum 1 Layer FWD] result: ")
    logger.info(s"Use java thread ${params.model} isNNRecurrent " +
      s"${model.isInstanceOf[nn.Recurrent[Float]]} " +
      s"isMKLDNNGRU ${model.isInstanceOf[RNN]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} ")
    logger.info(s"Average Throughput is: " +
      s"${params.batchSize.toDouble * params.iteration / (end3_inf - start3_inf) * 1e9} " +
      s"record / second.")

    println("========================================================================\n\n")

    gradOutputFormat = HeapData(Array(params.seqLength, params.batchSize,
      params.commonSize), Memory.Format.tnc)
    gradOutput_t = Tensor(gradOutputFormat.shape).rand(1.0, 1.0)

    direction = Direction.UnidirectionalLeft2Right
    val gru4_inf = RNN(AlgKind.VanillaGru, params.commonSize, params.commonSize,
      f, direction, layers = 5)

    val start4_inf = System.nanoTime()
    Engine.dnnComputing.invokeAndWait2((0 until 1).map(i => () => {
      gru4_inf.evaluate()
      gru4_inf.setRuntime(new MklDnnRuntime)
      gru4_inf.initFwdPrimitives(Array(inputFormat), InferencePhase)
      for (i <- 1 to params.iteration) {
        val output = gru4_inf.forward(input_t)
      }
    }))
    val end4_inf = System.nanoTime()

    logger.info("[Uni L2R 5 Layers FWD] result: ")
    logger.info(s"Use java thread ${params.model} isNNRecurrent" +
      s"${model.isInstanceOf[nn.Recurrent[Float]]} " +
      s"isMKLDNNGRU ${model.isInstanceOf[RNN]} engineType ${Engine.getEngineType()} " +
      s"batchSize ${params.batchSize} ")
    logger.info(s"Average Throughput is: " +
      s"${params.batchSize.toDouble * params.iteration / (end4_inf - start4_inf) * 1e9}" +
      s"record / second.")

    println("========================================================================\n\n")
  }

  def threadPredict(params: DistriPerfOriginParams, sc: SparkContext, modelLoad: Module[Float]):
  Unit = {
    println("inputSize = " + params.commonSize)
    println("hiddenSize = " + params.commonSize)
    println("batchSize = " + params.batchSize)
    println("seqLength = " + params.seqLength)
    println("iterations = " + params.iteration)
    println("engineType = " + params.engineType)

    //    if(params.engineType == "rnn_nn") {
    //      require(modelLoad.isInstanceOf[nn.Recurrent[Float]])
    //    }

    if(params.engineType == "rnn_mkldnn") {
      require(modelLoad.isInstanceOf[RNN])
    }

    var inputShape: Array[Int] = null
    var outputShape: Array[Int] = null

    params.engineType match {
      case "rnn_nn" =>
        inputShape = Array(params.batchSize, params.seqLength, params.commonSize)
        outputShape = Array(params.batchSize, params.seqLength, params.commonSize)
      case "rnn_mkldnn" =>
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
    println("fixed")

    //    MKL.setNumThreads(4)
    //    com.intel.analytics.bigdl.mkl.MklDnn.setNumThreads(4)
    //    Affinity.setOmpAffinity()


    parser.parse(argv, new DistriPerfOriginParams()).foreach { params =>
      if (params.engineType == "rnn_mkldnn") {
        System.setProperty("bigdl.engineType", "mkldnn")
        System.setProperty("bigdl.mklNumThreads", params.threadNum)
      }

      if (params.engineType == "rnn_nn") {
        System.setProperty("bigdl.engineType", "mklblas")
      }

      val conf = Engine.createSparkConf()
        .setAppName("Test perf")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)

      Engine.init
      Engine.setNodeAndCore(Engine.nodeNumber(), Engine.coreNumber())

      // NN GRU
      if(params.engineType == "rnn_nn") {
        println("blas")
        Engine.setEngineType(MklBlas)

        if(params.blasModelType == "unil2r1") {
          val nn_model = nn.Recurrent[Float]().add(nn.GRU(params.commonSize, params.commonSize))
          threadPredict(params, sc, nn_model)
          println("[Uni L2R 1 layer] result\n\n ")
        }

        if(params.blasModelType == "biconcat1") {
          val nn_model2 = nn.BiRecurrent[Float](nn.JoinTable[Float](3, 0)
            .asInstanceOf[AbstractModule[Table, Tensor[Float], Float]])
            .add(nn.GRU(params.commonSize, params.commonSize))

          threadPredict(params, sc, nn_model2)
          println("[Bi Concat 1 layer] result\n\n ")
        }

        if(params.blasModelType == "bisum1") {
          val nn_model3 = nn.BiRecurrent[Float](nn.CAddTable()
            .asInstanceOf[AbstractModule[Table, Tensor[Float], Float]])
            .add(nn.GRU(params.commonSize, params.commonSize))

          threadPredict(params, sc, nn_model3)
          println("[Bi Sum 1 layer] result\n\n ")
        }

        if(params.blasModelType == "unil2r5") {
          val nn_input = nn.Input()
          var nn_gru = nn.Recurrent[Float]()
            .add(nn.GRU(params.commonSize, params.commonSize)).inputs(nn_input)

          for (i <- 1 until 5) {
            nn_gru = nn.Recurrent[Float]()
              .add(nn.GRU(params.commonSize, params.commonSize)).inputs(nn_gru)
          }

          val nn_model4 = nn.Graph(nn_input, nn_gru)
          threadPredict(params, sc, nn_model4)
          println("[Uni L2R 5 layers] result\n\n ")
        }
      }

      // MKLDNN GRU
      if(params.engineType == "rnn_mkldnn") {
        println("\nmkldnn")
        println("Thread number: " + params.threadNum)

        val f = AlgKind.EltwiseTanh
        var direction = Direction.UnidirectionalLeft2Right

        val inputFormat = HeapData(Array(params.seqLength, params.batchSize,
          params.commonSize), Memory.Format.tnc)

        val gru1 = RNN(AlgKind.VanillaGru, params.commonSize, params.commonSize,
          f, direction, layers = 1)

        val modelLoad = gru1
        threadPredict(params, sc, modelLoad)
      }
    }
  }
}

case class DistriPerfOriginParams (
                              batchSize: Int = 20,
                              iteration: Int = 100,
                              commonSize: Int = 800,
                              seqLength: Int = 300,
                              threadNum: String = "1",
                              engineType: String = "rnn_mkldnn",
                              model: String = "vanilla_gru",
                              blasModelType: String = "unil2r",
                              threadPredict: Boolean = true
                            )


