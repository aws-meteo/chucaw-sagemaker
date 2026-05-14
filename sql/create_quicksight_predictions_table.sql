CREATE EXTERNAL TABLE IF NOT EXISTS silverlayer.downscaling_predictions (
  prediction_id string,
  model_name string,
  model_version string,
  prediction_timestamp_utc string,
  source_database string,
  source_table string,
  source_year string,
  source_month string,
  source_day string,
  source_hour string,
  run string,
  latitude double,
  longitude double,
  target_variable string,
  predicted_value double,
  predicted_units string,
  method string,
  observed_value double,
  error double,
  feature_snapshot_json string
)
PARTITIONED BY (
  year string,
  month string,
  day string,
  hour string,
  model_name_partition string,
  model_version_partition string
)
STORED AS PARQUET
LOCATION 's3://<your-bucket>/<your-prefix>/downscaling/predictions/'
TBLPROPERTIES ('parquet.compress'='SNAPPY');

-- Load partitions after writing new files:
-- MSCK REPAIR TABLE silverlayer.downscaling_predictions;
-- Or add partitions explicitly when needed:
-- ALTER TABLE silverlayer.downscaling_predictions ADD IF NOT EXISTS
-- PARTITION (year='2026',month='04',day='09',hour='18',model_name_partition='knn_baseline',model_version_partition='v1')
-- LOCATION 's3://<your-bucket>/<your-prefix>/downscaling/predictions/year=2026/month=04/day=09/hour=18/model_name_partition=knn_baseline/model_version_partition=v1/';
