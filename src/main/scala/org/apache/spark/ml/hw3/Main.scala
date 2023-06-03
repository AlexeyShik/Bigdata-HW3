package org.apache.spark.ml.hw3

import org.apache.spark.ml.linalg.{Vectors, Vector}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

object Main extends App {

  val spark = SparkSession.builder()
    .master("local[1]")
    .appName("hw3")
    .getOrCreate()

  import spark.sqlContext.implicits._

  val linearRegression: LinearRegression = new LinearRegression().setUseBatch(true)

  val data: DataFrame = spark.read
    .option("header", "true")
    .csv("./insurance.csv")
    .map(row => {
      Tuple2(
        Vectors.dense(
          java.lang.Double.parseDouble(row.getString(0)),
          if (row.getString(1) == "male") 1 else 0,
          if (row.getString(1) == "female") 1 else 0,
          java.lang.Double.parseDouble(row.getString(2)),
          java.lang.Double.parseDouble(row.getString(3)),
          if (row.getString(4) == "yes") 1 else 0,
          if (row.getString(4) == "no") 1 else 0,
          if (row.getString(5) == "southwest") 1 else 0,
          if (row.getString(5) == "northwest") 1 else 0,
          if (row.getString(5) == "southeast") 1 else 0,
          if (row.getString(5) == "northeast") 1 else 0,
        ),
        Vectors.dense(
          java.lang.Double.parseDouble(row.getString(6))
        )
      )
    }).toDF(linearRegression.getFeaturesCol, linearRegression.getLabelCol)

  val Array(dataTrain, dataTest) = data.randomSplit(Array[Double](0.8,  0.2), 1234)

  val model: LinearRegressionModel = linearRegression.fit(dataTrain)
  val labelsData: Dataset[Double] = dataTest.select(linearRegression.getLabelCol).map(row => row.get(0).asInstanceOf[Vector](0))
  val predictionsData: Dataset[Double] = model.transform(dataTest.drop(linearRegression.getLabelCol)).select(linearRegression.getPredictionCol).map(x => x.getDouble(0))

  val labels: Array[Double] = labelsData.collect()
  val predictions: Array[Double] = predictionsData.collect()

  val yMean: Double = labels.foldLeft(0.0)((a, b) => a + b) / labels.length
  var err2Sum: Double = 0.0
  var errSum: Double = 0.0
  var errBaseline: Double = 0.0
  for (i <- labels.indices) {
    err2Sum += Math.pow(labels(i) - predictions(i), 2)
    errSum += Math.abs(labels(i) - predictions(i))
    errBaseline += Math.pow(labels(i) - yMean, 2)
  }

  val R2 = 1.0 - err2Sum / errBaseline
  val MSE = err2Sum / labels.length
  val MAE = errSum / labels.length

  println("* R^2 = " + R2)
  println("* MSE = " + MSE)
  println("* MAE = " + MAE)
}
