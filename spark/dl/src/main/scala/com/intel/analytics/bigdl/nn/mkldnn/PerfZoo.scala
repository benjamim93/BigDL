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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, Sample}
import com.intel.analytics.bigdl.mkl.{MKL, Memory, MklDnn}
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.{Module, StaticGraph}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.nn.mkldnn.ResNet.DatasetType.ImageNet
import com.intel.analytics.bigdl.nn.mkldnn.models.Vgg_16
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Loss, Top1Accuracy, Top5Accuracy}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{Engine, T, Table, ThreadPool}
import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import scopt.OptionParser

import scala.reflect.ClassTag

object PerfZoo {

  val logger = Logger.getLogger(getClass)

  def main(argv: Array[String]): Unit = {
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("bigdl.engineType", "mkldnn")

    val conf = Engine.createSparkConf()
      .setAppName("testzoo")
      .setMaster("local[*]")
      .set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)

    Engine.init

    var model = Module.loadModule[Float](
       "/home/mengceng/zoo_models/imgcls/analytics-zoo_densenet-161_imagenet_0.1.0.model")

    /*
    val graph = model.toGraph().asInstanceOf[StaticGraph[Float]]
    val main_graph = graph.inputs(0).nextNodes(0).element.asInstanceOf[StaticGraph[Float]]

    val new_inputs = main_graph.inputs

    val main_graph_outputs = main_graph.outputs
    val new_outputs = graph.inputs(0).nextNodes(0).nextNodes(0)
    val main_graph_dummynode = main_graph_outputs(0).nextNodes(1)

    new_outputs.removePrevEdges()
    new_outputs.removeNextEdges()

    main_graph_dummynode.add(new_outputs)

    val new_graph = nn.Graph(new_inputs.toArray, new_outputs)
    new_graph.evaluate()
    new_graph.asInstanceOf[nn.StaticGraph[Float]]
      .setInputFormats(Seq(Memory.Format.nchw, Memory.Format.nc))
    new_graph.asInstanceOf[nn.StaticGraph[Float]].setOutputFormats(Seq(Memory.Format.nc))
    val model_ir = new_graph.asInstanceOf[nn.StaticGraph[Float]].toIRgraph()
    model_ir.evaluate()

    val num_channel = 3
    val img_size = 224

    val input = Tensor[Float](1, num_channel, img_size, img_size).rand()
    val input_imginfo = Tensor[Float](T(T(img_size.toFloat, img_size.toFloat, 1.0, 1.0)))

    val output = model_ir.forward(T(input, input_imginfo))
    println()
    */
    model.evaluate()
    model.asInstanceOf[nn.StaticGraph[Float]].setInputFormats(Seq(Memory.Format.nchw))
    model.asInstanceOf[nn.StaticGraph[Float]].setOutputFormats(Seq(Memory.Format.nchw))
//    val model_ir = model.asInstanceOf[nn.StaticGraph[Float]].toIRgraph()
//    model_ir.evaluate()

    val num_channel = 3
    val img_size = 224

    import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
    import com.intel.analytics.bigdl.transform.vision.image.ImageFeature

    // val input = Tensor[Float](1, num_channel, img_size, img_size).rand()
    //    val output = model.forward(input)

    import com.intel.analytics.bigdl.tensor.Tensor
    val imf = ImageFeature()
    imf(ImageFeature.imageTensor) = Tensor[Float](Array(3, 224, 224)).rand()
    val input = ImageFrame.array(Array(imf))
    //    val file = new File("/home/mengceng/zoo_imgs/image.jpeg")
    //    val input = ImageFrame.read("/home/mengceng/zoo_imgs/image.jpeg", sc, 1)
    val output = model.predictImage(input)

    println()
  }
}