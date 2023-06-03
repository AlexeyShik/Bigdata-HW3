package org.apache.spark.ml.hw3

import breeze.linalg.{DenseMatrix, Matrix}
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{BooleanParam, ParamMap}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, PredictionModel, PredictorParams}
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.sql.types.StructType

trait LinearRegressionParams extends PredictorParams {

  val useBatch = new BooleanParam(this, "useBatch", "Whenever to use batch calculation")
  def isUseBatch : Boolean = $(useBatch)
  def setUseBatch(value: Boolean) : this.type = set(useBatch, value)
  setDefault(useBatch -> false)
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    implicit val encoder : Encoder[Vector] = ExpressionEncoder()

    val features: Dataset[Vector] = dataset.select(dataset($(featuresCol)).as[Vector])
    val labels: Dataset[Vector] = dataset.select(dataset($(labelCol)).as[Vector])
    val data = features
      .withColumn("id", monotonically_increasing_id())
      .join(labels.withColumn("id", monotonically_increasing_id()),  "id")
      .drop("id")
    val n: Int = features.first().size
    val m: Long = features.count()
    var w: breeze.linalg.Vector[Double] = Vectors.zeros(n).asBreeze

    for (i <- 0 until n) {
      w(i) = 1.0 / n
    }
    val meanLabel = labels.reduce((v1, v2) => Vectors.dense(v1(0) + v2(0)))(0) / labels.count()

    val eps: Double = 1e-7
    val lambda: Double = 1.0 / m / Math.max(1,  meanLabel)
    var error: Double = 1e9
    var maxIterations: Int = 100000

    while (Math.abs(error) > eps && maxIterations > 0) {

      val mapFunc = if (isUseBatch) createBatchMapFunc(n, lambda, w) else createMapFunc(n, lambda, w)
      val reduceFunc = createReduceFunc()
      val (gradientSummary, errorSummary): (breeze.linalg.Vector[Double], Double) = data.rdd
        .mapPartitions(mapFunc)
        .reduce(reduceFunc)

      error = errorSummary / m
      w = w - gradientSummary
      maxIterations -= 1
    }

    copyValues(new LinearRegressionModel(Vectors.fromBreeze(w))).setParent(this)
  }

  private def createMapFunc(n: Int, lambda: Double, w: breeze.linalg.Vector[Double]): Iterator[Row] => Iterator[(breeze.linalg.Vector[Double], Double)] = {
    (data: Iterator[Row]) => {
      val initialValue: (breeze.linalg.Vector[Double], Double) = (Vectors.zeros(n).asBreeze, 0.0)

      val foldFunc = (tuple: (breeze.linalg.Vector[Double], Double), row: Row) => {
        val x: breeze.linalg.Vector[Double] = row.get(0).asInstanceOf[Vector].asBreeze
        val y: Double = row.get(1).asInstanceOf[Vector](0)
        val prediction: Double = w.dot(x)
        val error: Double = y - prediction
        val gradient = tuple._1
        val errorSum = tuple._2 + error

        for (i <- 0 until x.size) {
          gradient(i) -= 2 * lambda * error * x(i)
        }

        (gradient, errorSum)
      }

      val (gradientPart, errorPart): (breeze.linalg.Vector[Double], Double) = data.foldLeft(initialValue)(foldFunc)
      Iterator(Tuple2(gradientPart, errorPart))
    }
  }

  private def createBatchMapFunc(n: Int, lambda: Double, w: breeze.linalg.Vector[Double]): Iterator[Row] => Iterator[(breeze.linalg.Vector[Double], Double)] = {
    (data: Iterator[Row]) => {
      val rows: Array[Row] = data.map(x => x).toArray
      val xs: Array[breeze.linalg.Vector[Double]] = rows.map(row => row.get(0).asInstanceOf[Vector].asBreeze)
      val Y: breeze.linalg.Vector[Double] = Vectors.dense(rows.map(row => row.get(1).asInstanceOf[Vector](0))).asBreeze

      val X: Matrix[Double] = DenseMatrix.zeros(Y.size, n)
      for (i <- 0 until Y.size) {
        for (j <- 0 until n) {
          X(i, j) = xs(i)(j)
        }
      }

      val prediction: breeze.linalg.Vector[Double] = X * w

      val gradient: breeze.linalg.Vector[Double] = Vectors.zeros(n).asBreeze
      var errorSum: Double = 0

      for (i <- 0 until Y.size) {
        val error = Y(i) - prediction(i)
        errorSum += error

        for (j <- 0 until n) {
          gradient(j) -= 2 * lambda * error * X(i, j)
        }
      }

      Iterator(Tuple2(gradient, errorSum))
    }
  }

  private def createReduceFunc(): ((breeze.linalg.Vector[Double], Double), (breeze.linalg.Vector[Double], Double)) => (breeze.linalg.Vector[Double], Double) = {
    (accumulator, current) => (accumulator._1 + current._1, accumulator._2 + current._2)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    if (schema.fieldNames.contains($(labelCol))) {
      SchemaUtils.checkColumnType(schema, getLabelCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getLabelCol).copy(name = getLabelCol))
    }
  }

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[hw3](override val uid: String, val w: DenseVector) extends PredictionModel[Vector, LinearRegressionModel] with LinearRegressionParams with MLWritable {

  private[hw3] def this(w: Vector) = this(Identifiable.randomUID("linearRegressionModel"), w.toDense)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(w), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform", (x: Vector) => w.dot(x))
    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }

  override def predict(features: Vector): Double = w.dot(features)

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getPredictionCol).copy(name = getPredictionCol))
    }
  }

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      sqlContext.createDataFrame(Seq(Tuple1(w.toDense))).write.parquet(path + "/weights")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/weights")

      implicit val encoder : Encoder[Vector] = ExpressionEncoder()

      val weights =  vectors.select(vectors("_1").as[Vector]).first()

      val model = new LinearRegressionModel(weights)
      metadata.getAndSetParams(model)
      model
    }
  }
}