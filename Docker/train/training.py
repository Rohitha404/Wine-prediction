import sys
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def transform_data(dataset):
    return dataset.select(*(col(field).cast("double").alias(field.strip("\"")) for field in dataset.columns))

if __name__ == "__main__":
    print("Initializing Spark Application...")

    spark = SparkSession.builder.appName("WineQualityPredictionApp").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    spark._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    input_path = "TrainingDataset.csv"
    model_path = "/job/trainedmodel"

    print(f"Loading training data from {input_path}")
    raw_df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("sep", ";")
        .option("inferschema", "true")
        .load(input_path)
    )
    
    prepared_data = transform_data(raw_df)

    features = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'quality'
    ]

    print("Configuring VectorAssembler...")
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    
    print("Setting up StringIndexer...")
    indexer = StringIndexer(inputCol="quality", outputCol="label")

    print("Persisting the dataset for performance improvement...")
    prepared_data.cache()
    
    print("Building RandomForestClassifier...")
    classifier = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=150,
        maxDepth=15,
        seed=150,
        impurity="gini"
    )
    
    print("Creating training pipeline...")
    pipeline = Pipeline(stages=[assembler, indexer, classifier])
    pipeline_model = pipeline.fit(prepared_data)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    print("Tuning model parameters with CrossValidator...")
    param_grid = (
        ParamGridBuilder()
        .addGrid(classifier.maxDepth, [5, 10])
        .addGrid(classifier.numTrees, [50, 150])
        .addGrid(classifier.minInstancesPerNode, [5])
        .addGrid(classifier.seed, [100, 200])
        .addGrid(classifier.impurity, ["entropy", "gini"])
        .build()
    )
    
    cross_validator = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=2
    )

    print("Training with CrossValidator...")
    cv_model = cross_validator.fit(prepared_data)
    
    optimal_model = cv_model.bestModel

    print("Saving the optimal model to disk...")
    optimal_model.write().overwrite().save(model_path)
    spark.stop()
