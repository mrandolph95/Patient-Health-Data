# Databricks notebook source
# MAGIC %md
# MAGIC ### Create a Posteoperative Patient Data Predictive Model

# COMMAND ----------

# MAGIC %md
# MAGIC #### Packages

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import when, col
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, NotebookTask, Source

# COMMAND ----------

# MAGIC %md
# MAGIC #### Mount My Storage to db

# COMMAND ----------

storage_account_name = "mlrstorage2025"
container_name = "my-container"
mount_point = "/mnt/postoperative_data"
sas_token = "sp=r&st=2025-03-26T05:37:45Z&se=2025-03-26T13:37:45Z&spr=https&sv=2024-11-04&sr=c&sig=bU4ljnmAWDEJyDp2gc2DVPCaFFQbxcfOjicy5sFeLcQ%3D"

configs = {
    f"fs.azure.sas.{container_name}.{storage_account_name}.blob.core.windows.net": sas_token
}

dbutils.fs.mount(
  source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net",
  mount_point = mount_point,
  extra_configs = configs
)

# COMMAND ----------

dbutils.fs.mounts()

# COMMAND ----------

file_path = "/mnt/postoperative_data/postoperative_data.parquet"
df = spark.read.parquet(file_path)

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Process Data using Spark

# COMMAND ----------

df = df.withColumn("index", monotonically_increasing_id())
df = df.select("index", "*")
display(df)

# COMMAND ----------

# Categorial feature transformation
indexer = StringIndexer(inputCol="L-CORE", outputCol="L-CORE_index")
pipeline = Pipeline(stages=[indexer])
df = pipeline.fit(df).transform(df)
display(df)

# COMMAND ----------

df = df.withColumn(
    "COMFORT_Category",
    when(col("COMFORT").between(0, 5), "Very Uncomfortable")
    .when(col("COMFORT").between(6, 10), "Uncomfortable")
    .when(col("COMFORT").between(11, 15), "Comfortable")
    .when(col("COMFORT").between(16, 20), "Very Comfortable")
    .otherwise("Unknown")
)

display(df)

# COMMAND ----------

column_mapping = {
    ('L-CORE', 'L_CORE'), 
    ('L-SURF', 'L_SURF'), 
    ('L-O2', 'L_O2'), 
    ('L-BP', 'L_BP'), 
    ('SURF-STBL', 'SURF_STBL'), 
    ('CORE-STBL', 'CORE_STBL'), 
    ('BP-STBL', 'BP_STBL'),
    ('decision', 'discharge_decision')
}

for old_name, new_name in column_mapping:
    df = df.withColumnRenamed(old_name, new_name)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train Model

# COMMAND ----------

feature_columns = ["L_CORE","L_SURF","L_O2","L_BP","SURF_STBL","CORE_STBL","BP_STBL","COMFORT","discharge_decision","COMFORT_Category"]

# COMMAND ----------

indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in feature_columns]
encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_encoded") for col in feature_columns]

for indexer, encoder in zip(indexers, encoders):
    df = indexer.fit(df).transform(df)
    df = encoder.fit(df).transform(df)

encoded_columns = [col + "_encoded" for col in feature_columns]
feature_columns = [col for col in feature_columns if col not in feature_columns] + encoded_columns

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df = assembler.transform(df)

display(df)

# COMMAND ----------

# Split the data
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train the model
classifier = RandomForestClassifier(labelCol="discharge_decision_index", featuresCol="features")
model = classifier.fit(train_data)

predictions = model.transform(test_data)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="discharge_decision_index", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)

print(f"Model accuracy: {accuracy}")

# COMMAND ----------

