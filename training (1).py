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

def clean_data(data_frame):
    return data_frame.select(*(col(column).cast("double").alias(column.strip("\"")) for column in data_frame.columns))

if __name__ == "__main__":
    print("Start the Spark App")


    spark_session = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

    spark_context = spark_session.sparkContext
    spark_context.setLogLevel('ERROR')

    spark_session._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")


    training_data_path = "s3://predofwine/TrainingDataset.csv"
    model_output_path = "s3://predofwine/trainedmodel"

    print(f"Read the training CSV file from {training_data_path}")
    raw_data_frame = (spark_session.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema", 'true')
          .load(training_data_path))
    
    training_data_frame = clean_data(raw_data_frame)

    feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                    'pH', 'sulphates', 'alcohol', 'quality']

    print("Construct the VectorAssembler")
    features_assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
    
    print("Constructing the StringIndexer")
    label_indexer = StringIndexer(inputCol="quality", outputCol="label")

    print("Cache the data")
    training_data_frame.cache()
    
    print("Constructing the RandomForestClassifier")
    random_forest_classifier = RandomForestClassifier(labelCol='label', 
                                                      featuresCol='features',
                                                      numTrees=150,
                                                      maxDepth=15,
                                                      seed=150,
                                                      impurity='gini')
    
    print("Constructing the pipeline for training")
    training_pipeline = Pipeline(stages=[features_assembler, label_indexer, random_forest_classifier])
    fitted_model = training_pipeline.fit(training_data_frame)

    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol='label', 
                                                           predictionCol='prediction', 
                                                           metricName='accuracy')

    print("Retrain the model")
    cv_model = None
    parameter_grid = ParamGridBuilder() \
        .addGrid(random_forest_classifier.maxDepth, [5, 10]) \
        .addGrid(random_forest_classifier.numTrees, [50, 150]) \
        .addGrid(random_forest_classifier.minInstancesPerNode, [5]) \
        .addGrid(random_forest_classifier.seed, [100, 200]) \
        .addGrid(random_forest_classifier.impurity, ["entropy", "gini"]) \
        .build()
    
    cv_pipeline = CrossValidator(estimator=training_pipeline,
                                 estimatorParamMaps=parameter_grid,
                                 evaluator=accuracy_evaluator,
                                 numFolds=2)

    print("Fit the CrossValidator")
    best_model = cv_pipeline.fit(training_data_frame)
    
    final_model = best_model.bestModel

    print("Save model to S3")
    final_model_path = model_output_path
    final_model.write().overwrite().save(final_model_path)
    spark_session.stop()
