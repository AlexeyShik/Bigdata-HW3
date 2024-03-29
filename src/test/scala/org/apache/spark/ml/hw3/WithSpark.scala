package org.apache.spark.ml.hw3

import org.apache.spark.sql.{SQLContext, SparkSession}

trait WithSpark {
  lazy val spark: SparkSession = WithSpark._spark
  lazy val sqlc: SQLContext = WithSpark._sqlc
}

object WithSpark {
  lazy val _spark: SparkSession = SparkSession.builder
    .appName("Simple Application")
    .master("local[4]")
    .getOrCreate()

  lazy val _sqlc: SQLContext = _spark.sqlContext
}
