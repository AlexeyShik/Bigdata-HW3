package org.apache.spark.ml.hw3

import com.google.common.io.Files
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should

import scala.util.Random

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val deltaSmall = 1e-5
  val deltaBig = 1e-1

  lazy val dataSimple: DataFrame = LinearRegressionTest._dataSimple
  lazy val featuresSimple: DataFrame = LinearRegressionTest._featuresSimple
  lazy val labelsSimple: Seq[Double] = LinearRegressionTest._labelsSimple

  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val features: DataFrame = LinearRegressionTest._features
  lazy val labels: Seq[Double] = LinearRegressionTest._labels

  "Estimator" should "calculate weights, easy sample" in {
    val estimator = new LinearRegression()
    val model = estimator.fit(dataSimple)

    model.w(0) should be(1.0 +- deltaSmall)
  }

  "Estimator" should "calculate weights, batch mode, easy sample" in {
    val estimator = new LinearRegression().setUseBatch(true)
    val model = estimator.fit(dataSimple)

    model.w(0) should be(1.0 +- deltaSmall)
  }

  "Model" should "predict value, easy sample" in {
    val model: LinearRegressionModel = new LinearRegressionModel(w = Vectors.dense(1.0).toDense)

    validateModel(model)
  }

  "Estimator" should "calculate weights, normal sample" in {
    val estimator = new LinearRegression()
    val model = estimator.fit(data)

    model.w(0) should be(1.5 +- deltaBig)
    model.w(1) should be(0.3 +- deltaBig)
    model.w(2) should be(-0.7 +- deltaBig)
  }

  "Estimator" should "calculate weights, batch mode, normal sample" in {
    val estimator = new LinearRegression().setUseBatch(true)
    val model = estimator.fit(data)

    model.w(0) should be(1.5 +- deltaBig)
    model.w(1) should be(0.3 +- deltaBig)
    model.w(2) should be(-0.7 +- deltaBig)
  }

  "Model" should "predict value, normal sample" in {
    val model: LinearRegressionModel = new LinearRegressionModel(w = Vectors.dense(1.5, 0.3, -0.7).toDense)

    val result: Array[Double] = model.transform(features).collect().map(x => x.getAs[Double]("prediction"))
    val labelsIt: Iterator[Double] = labels.iterator
    val resultIt: Iterator[Double] = result.iterator
    while (labelsIt.hasNext) {
      resultIt.next() should be (labelsIt.next() +- deltaSmall)
    }
    resultIt.hasNext should be (false)
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(new LinearRegression()))

    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)
    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)
    val model = reRead.fit(dataSimple).stages(0).asInstanceOf[LinearRegressionModel]

    model.w(0) should be(1.0 +- deltaSmall)
  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(new LinearRegression()))

    val model = pipeline.fit(dataSimple)
    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)
    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModel(reRead.stages(0).asInstanceOf[LinearRegressionModel])
  }

  private def validateModel(model: LinearRegressionModel): Unit = {
    val result: Array[Double] = model.transform(featuresSimple).collect().map(x => x.getAs[Double]("prediction"))
    val labelsIt: Iterator[Double] = labelsSimple.iterator
    val resultIt: Iterator[Double] = result.iterator
    while (labelsIt.hasNext) {
      resultIt.next() should be (labelsIt.next() +- deltaSmall)
    }
    resultIt.hasNext should be (false)
  }

}

object LinearRegressionTest extends WithSpark {

  lazy val _dataSimple: DataFrame = {
    import sqlc.implicits._
    Seq(
      (Vectors.dense(1.0), Vectors.dense(1.0)),
      (Vectors.dense(2.0), Vectors.dense(2.0)),
      (Vectors.dense(3.0), Vectors.dense(3.0))
    ).map(x => Tuple2(x._1, x._2)).toDF("features", "label")
  }

  lazy val _featuresSimple: DataFrame = {
    import sqlc.implicits._
    Seq(
      Vectors.dense(4.0),
      Vectors.dense(5.0),
      Vectors.dense(6.0)
    ).map(x => Tuple1(x)) toDF "features"
  }

  lazy val _labelsSimple: Seq[Double] = {
    Seq(4.0, 5.0, 6.0)
  }

  lazy val _rows: Array[(Vector, Vector)] = {
    val size: Int = 100000
    val rows: Array[(Vector, Vector)] = new Array[(Vector, Vector)](size)
    val random: Random = new Random(1234567)
    for (i <- 0 until size) {
      val noise: Double = random.nextDouble()
      val x1: Double = random.nextDouble()
      val x2: Double = random.nextDouble()
      val x3: Double = random.nextDouble()
      val y: Double = 1.5 * x1 + 0.3 * x2 - 0.7 * x3 + noise * 1e-7
      rows(i) = (Vectors.dense(x1, x2, x3), Vectors.dense(y))
    }
    rows
  }

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    _rows.toSeq.map(row => Tuple2(row._1, row._2)).toDF("features",  "label")
  }

  lazy val _features: DataFrame = {
    _data.select("features")
  }

  lazy val _labels: Seq[Double] = {
    _data.select("label").collect().map(x => x.get(0).asInstanceOf[Vector](0)).toSeq
  }
}

