from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
from pyspark.sql.functions import *
inputFile="Chicago_Crime_Without_Location.csv"

#data
df=spark.read.option("header", True).option("delimiter", ";").csv(inputFile)
df=df.withColumnRenamed("Primary Type","type")\
            .withColumnRenamed("Location Description","location")
df=df.select(df.ID.cast('int').alias("id"),\
             df.Beat.alias("beat"),\
             df.type,\
             #df.Description.alias('description'),\
             df.location,\
             df.Arrest.cast('Boolean').alias("arrest"),\
             to_timestamp(df.Date, 'MM/dd/yy hh:mm a').alias('datetime'))


from pyspark.ml import Pipeline
from pyspark.ml.feature import FeatureHasher
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

mldf=df.withColumn('year',year('datetime'))\
    .withColumn('month',month('datetime'))\
    .withColumn('day',dayofmonth('datetime'))\
    .withColumn('hour',hour('datetime'))\
    .withColumn('label',df.arrest.cast('int'))\
    .drop('datetime','id','arrest')
mldf.printSchema()
training, test=mldf.randomSplit([0.7,0.3])
hasher = FeatureHasher(inputCols=['beat', 'type', 'location'],\
                       categoricalCols=['year','month','day','hour'],\
                       outputCol="features")
nb = NaiveBayes()
paramGrid = ParamGridBuilder()\
            .addGrid(nb.smoothing, [0.5, 1.0, 2.0])\
            .build()
nb.explainParam('smoothing')
nbPipeline=Pipeline(stages=[hasher,nb])
crossval=CrossValidator(estimator=nbPipeline,\
                          estimatorParamMaps=paramGrid,\
                          evaluator=BinaryClassificationEvaluator(),\
                          numFolds=3)
cvModel=crossval.fit(training)
#cvModel.bestModel.write().overwrite().save('model/nb')
predictions = cvModel.transform(test)
#predictions = predictions.select("label", "features", "probability", "prediction")
predictions.show()
nbModel=cvModel.bestModel.stages[1]
#nbTrainingSummary = nbModel.summary
AUC=BinaryClassificationEvaluator().evaluate(predictions)
print("Test evaluation:\narea under ROC: %s"%(str(AUC)))