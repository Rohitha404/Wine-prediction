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

def process_data(input_frame):
    return input_frame.select(*(col(field).cast("double").alias(field.strip("\"")) for field in input_frame.columns))

if __name__ == "__main__":
    print("Launching Spark Application...")

    session = SparkSession.builder.appName("WineQualityPredictionApp").getOrCreate()
    context = session.sparkContext
    context.setLogLevel('ERROR')

    session._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    input_file = "TrainingDataset.csv"
    model_output_path = "/job/trainedmodel"

    print(f"Loading dataset from {input_file}")
    raw_frame = (
        session.read.format("csv")
        .option("header", "true")
        .option("sep", ";")
        .option("inferschema", "true")
        .load(input_file)
    )
    
    cleaned_data = process_data(raw_frame)

    selected_features = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'quality'
    ]

    print("Initializing VectorAssembler...")
    assembler = VectorAssembler(inputCols=selected_features, outputCol="features")
    
    print("Setting up StringIndexer...")
    indexer = StringIndexer(inputCol="quality", outputCol="label")

    print("Caching the processed dataset for optimized performance...")
    cleaned_data.cache()
    
    print("Configuring RandomForestClassifier...")
    classifier = RandomForestClassifier(
        labelCol="label", 
        featuresCol="features", 
        numTrees=150, 
        maxDepth=15, 
        seed=150, 
        impurity="gini"
    )
    
    print("Defining the training pipeline...")
    pipeline = Pipeline(stages=[assembler, indexer, classifier])
    trained_pipeline = pipeline.fit(cleaned_data)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", 
        predictionCol="prediction", 
        metricName="accuracy"
    )

    print("Setting up CrossValidator for parameter tuning...")
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

    print("Training the model using CrossValidator...")
    cross_validated_model = cross_validator.fit(cleaned_data)
    
    optimal_model = cross_validated_model.bestModel

    print("Storing the trained model to disk...")
    optimal_model.write().overwrite().save(model_output_path)
    session.stop()
