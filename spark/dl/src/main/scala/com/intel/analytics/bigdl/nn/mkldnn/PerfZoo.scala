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
import com.intel.analytics.bigdl.nn.Module
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

    /*
    val conf = Engine.createSparkConf()
      .setAppName("testzoo")
      .setMaster("local[*]")
      .set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)
    */

    Engine.init

    val model = Module.loadModule[Float](
      "/home/mengceng/bigdl_models/imgcls/vgg-19_imagenet")
    model.asInstanceOf[nn.StaticGraph[Float]].setInputFormats(Seq(Memory.Format.nchw))
    model.asInstanceOf[nn.StaticGraph[Float]].setOutputFormats(Seq(Memory.Format.nc))
    val model_ir = model.asInstanceOf[nn.StaticGraph[Float]].toIRgraph()
    model_ir.evaluate()

    val num_channel = 3
    val img_size = 224

    val input = Tensor[Float](1, num_channel, img_size, img_size).fill(0.8f)
    /*
    val label = Tensor[Float](1).fill(1.0f)
    val data = Sample(input, label)

    val dataset = new Array[Sample[Float]](1)
    dataset(0) = Sample(input, label)

    val dataSet = DataSet.array(dataset, sc).toDistributed().data(train = false)

    val output = model_ir.evaluate(dataSet,
      Array(new Top1Accuracy[Float](),
            new Top5Accuracy[Float](),
            new Loss[Float](CrossEntropyCriterion[Float]())))
      }
    */

    val output = model_ir.forward(input)

    println()
  }
}