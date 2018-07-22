from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
from pyspark.sql.functions import *
inputFile="ProjectData/Chicago_Crime_Without_Location.csv"

#data
df= spark.read.option("header", True).option("delimiter", ";").csv(inputFile)
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
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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
#lr=LogisticRegression(maxIter=10,family='binomial')
dt=DecisionTreeClassifier()
paramGrid = ParamGridBuilder()\
            .addGrid(dt.maxDepth, [5, 10])\
            .build()
dt.explainParam('maxDepth')
dtPipeline=Pipeline(stages=[hasher,dt])
crossval=CrossValidator(estimator=dtPipeline,\
                          estimatorParamMaps=paramGrid,\
                          evaluator=MulticlassClassificationEvaluator(),\
                          numFolds=5)
cvModel=crossval.fit(training)
predictions = cvModel.transform(test)
predictions = predictions.select("beat", "feature", "probability", "prediction")
predictions.show()
dtModel=cvModel.estimator.stages[1]
dtTrainingSummary = dtModel.summary
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy: %s"%(accuracy))