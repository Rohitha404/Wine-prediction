import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def clean_data(data_frame):
    """Cleans data by casting columns to double and stripping extra quotes."""
    return data_frame.select(*(col(column).cast("double").alias(column.strip("\"")) for column in data_frame.columns))

if __name__ == "__main__":
    print("Start the Spark App")

    spark_session = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()
    spark_context = spark_session.sparkContext
    spark_context.setLogLevel('ERROR')

   
    spark_context._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    
    validation_data_path = str(sys.argv[1])
    trained_model_path = "s3://wineprecdit/trainedmodel"

    
    raw_data_frame = (spark_session.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema", 'true')
          .load(validation_data_path))

    
    clean_data_frame = clean_data(raw_data_frame)

    
    prediction_model = PipelineModel.load(trained_model_path)

    
    predictions = prediction_model.transform(clean_data_frame)

    
    prediction_results = predictions.select(['prediction', 'label'])
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
    accuracy = accuracy_evaluator.evaluate(predictions)
    print(f'Test Accuracy = {accuracy}')

    
    prediction_metrics = MulticlassMetrics(prediction_results.rdd.map(tuple))
    weighted_f1_score = prediction_metrics.weightedFMeasure()
    print(f'Weighted F1 Score = {weighted_f1_score}')

    print("Exit the Spark App")
    spark_session.stop()
