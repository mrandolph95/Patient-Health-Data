# Postoperative Patient Data Pipeline

## Overview
This project implements a data pipeline using Apache Spark in Databricks to ingest, transform, and store postoperative patient data. The pipeline processes structured data stored in Azure Blob Storage, applies transformations, and prepares it for downstream analytics and modeling.

## Workflow
1. **Data Ingestion**
   - Mount Azure Blob Storage in Databricks
   - Load raw data in Parquet format
2. **Data Transformation**
   - Assign unique indexes to records
   - Standardize column names
   - Encode categorical variables using `StringIndexer` and `OneHotEncoder`
   - Aggregate features for analytics
3. **Data Storage & Management**
   - Write processed data back to Azure Blob Storage
   - Ensure efficient query execution for future use cases

## Technologies Used
- **Apache Spark** (PySpark for distributed data processing)
- **Databricks** (for scalable ETL execution)
- **Azure Blob Storage** (for cloud-based data storage)
- **Python** (for scripting and automation)

## Setup Instructions

### 1. Mount Azure Blob Storage
Ensure that your storage account and container are correctly configured before running the mount command:
```python
storage_account_name = "mlrstorage2025"
container_name = "my-container"
mount_point = "/mnt/postoperative_data"
sas_token = "your_sas_token"

configs = {
    f"fs.azure.sas.{container_name}.{storage_account_name}.blob.core.windows.net": sas_token
}

dbutils.fs.mount(
  source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net",
  mount_point = mount_point,
  extra_configs = configs
)
```

### 2. Load and Process Data
Load the dataset stored in Parquet format:
```python
file_path = "/mnt/postoperative_data/postoperative_data.parquet"
df = spark.read.parquet(file_path)
display(df)
```

### 3. Data Transformation
- Assign unique index to rows
- Standardize column names for consistency
- Encode categorical variables for structured storage
- Aggregate and prepare data for analytics

### 4. Store Processed Data
Write the transformed DataFrame back to Azure Blob Storage:
```python
output_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/your/output/path/"
df_without_index.write.mode("overwrite").parquet(output_path)
```

## Data Source
Postoperative Patient Data: [OpenML Dataset #34](https://www.openml.org/search?type=data&status=active&tags=clinical&sort=runs&id=34)

## Future Enhancements
- Automate pipeline execution using Databricks Workflows
- Integrate with Delta Lake for versioning and schema evolution
- Implement monitoring/logging for data quality checks
