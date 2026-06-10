## 1. Eksplorasi Dataset
Berdasarkan hasil analisis eksploratif (EDA) pada *raw data*, berikut adalah karakteristik dataset yang digunakan:
* **Target Prediksi (Label):** `total load actual` (Beban listrik aktual dalam Megawatt/MW). Distribusi target ini tergolong stabil dan bersih tanpa *outlier* ekstrem.
* **Fitur Prediktor Numerik (Cuaca & Waktu):** `temp` (suhu), `pressure` (tekanan udara), `humidity`, `wind_speed`, serta fitur waktu yang diekstrak (`hour`, `day_of_week`, `is_weekend`). 
    * *Insight Preprocessing:* Pada kolom `pressure`, terdeteksi >11.000 baris (6.6%) anomali/error sensor. Oleh karena itu, diterapkan pembersihan data menggunakan metode IQR (*Interquartile Range*) di **Silver Layer**.
* **Fitur Prediktor Kategorik (Spasial/Lokasi):** `city_name` (Madrid, Bilbao, Seville, Barcelona, Valencia). 
    * *Insight Preprocessing:* Karena data kota berupa teks, metode *Korelasi Pearson tidak dapat digunakan*. Sebagai gantinya, fitur kota diubah menjadi matriks angka menggunakan **One-Hot Encoding**, lalu dievaluasi tingkat signifikansinya menggunakan algoritma bawaan **Decision Tree (Feature Importance)** di **Gold Layer**.
* **Data Kosong (Null):** Terdapat dua kolom dari Kaggle yang 100% kosong (`generation hydro pumped storage aggregated` dan `forecast wind offshore eday ahead`). Kolom ini diabaikan saat pembentukan *dataset Machine Learning* di Gold Layer agar tidak merusak proses *dropna()*.
---


## 2. Set-Up Infrastruktur (MinIO & Apache Spark)

### 2.1 Buat Struktur Folder & Unduh Dataset
```bash
mkdir -p ~/tubes_k11/data/raw-data
mkdir -p ~/tubes_k11/scripts
cd ~/tubes_k11

# Unduh dataset raw
cd ~/tubes_k11/data/raw-data
wget [https://raw.githubusercontent.com/AliAristoMuthahhariParisi/ABD-TUBES/refs/heads/main/Dataset/energy_dataset.csv](https://raw.githubusercontent.com/AliAristoMuthahhariParisi/ABD-TUBES/refs/heads/main/Dataset/energy_dataset.csv)
wget [https://raw.githubusercontent.com/AliAristoMuthahhariParisi/ABD-TUBES/refs/heads/main/Dataset/weather_features.csv](https://raw.githubusercontent.com/AliAristoMuthahhariParisi/ABD-TUBES/refs/heads/main/Dataset/weather_features.csv)

# Unduh instalasi Apache Spark
cd ~/tubes_k11
wget [https://archive.apache.org/dist/spark/spark-3.5.5/spark-3.5.5-bin-hadoop3.tgz](https://archive.apache.org/dist/spark/spark-3.5.5/spark-3.5.5-bin-hadoop3.tgz)
```

### 2.2 Buat Dockerfile
```bash
nano ~/tubes_k11/Dockerfile
```
Isi dengan:
```bash
FROM python:3.11-slim

RUN apt-get update && apt-get install -y default-jdk-headless procps curl && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

COPY spark-3.5.5-bin-hadoop3.tgz /opt/
RUN tar -xzf /opt/spark-3.5.5-bin-hadoop3.tgz -C /opt/ \
    && mv /opt/spark-3.5.5-bin-hadoop3 /opt/spark \
    && rm /opt/spark-3.5.5-bin-hadoop3.tgz

ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin
ENV PYSPARK_PYTHON=python3

RUN pip install --no-cache-dir pyspark==3.5.5 pandas matplotlib seaborn boto3
WORKDIR /app
```
### 2.3 Buat docker-compose.yml
```bash
nano ~/tubes_k11/docker-compose.yml
```
Isi dengan:
```bash
version: '3.8'
services:
  minio:
    image: minio/minio:latest
    container_name: tubes-k11-minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: admin123
    volumes:
      - ./data:/data
    command: server /data --console-address ":9001"

  spark:
    image: tubes-k11-spark:3.5.5
    container_name: tubes-k11-spark
    ports:
      - "4040:4040"
    volumes:
      - ./scripts:/app/scripts
      - ./data:/data
    stdin_open: true
    tty: true
    depends_on:
      - minio
```

### 2.4 Build Image & Jalankan Container
```bash
cd ~/tubes_k11
docker build -t tubes-k11-spark:3.5.5 .
docker compose up -d
```


## 3. Bronze Layer (MinIo Upload)
Buka browser di Windows, masuk ke http://localhost:9001 (Login: admin / admin123).
Lalu buat 3 buah bucket: bronze, silver, gold.
Serta jalankan perintah berikut di terminal:
```bash
cd ~/tubes_k11
wget [https://dl.min.io/client/mc/release/linux-amd64/mc](https://dl.min.io/client/mc/release/linux-amd64/mc)
chmod +x mc
sudo mv mc /usr/local/bin/

mc alias set local http://localhost:9000 admin admin123
mc cp ~/tubes_k11/data/raw-data/energy_dataset.csv local/bronze/
mc cp ~/tubes_k11/data/raw-data/weather_features.csv local/bronze/
```


## 4. Silver Layer (Cleaning & IQR Outlier)
```bash
nano ~/tubes_k11/scripts/02_silver.py
```
Isi dengan:
```bash
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("tubes_sains_data_silver").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df_energy = spark.read.option("header", "true").option("inferSchema", "true").csv("/data/raw-data/energy_dataset.csv")
df_weather = spark.read.option("header", "true").option("inferSchema", "true").csv("/data/raw-data/weather_features.csv")

# 1. FORWARD FILL (Missing Values)
window_energy = Window.orderBy("time").rowsBetween(Window.unboundedPreceding, 0)
numeric_energy = [c for c, t in df_energy.dtypes if t in ("double", "float", "int", "bigint")]
for c in numeric_energy:
    df_energy = df_energy.withColumn(c, F.last(F.col(c), ignorenulls=True).over(window_energy))

window_weather = Window.orderBy("dt_iso").rowsBetween(Window.unboundedPreceding, 0)
numeric_weather = [c for c, t in df_weather.dtypes if t in ("double", "float", "int", "bigint")]
for c in numeric_weather:
    df_weather = df_weather.withColumn(c, F.last(F.col(c), ignorenulls=True).over(window_weather))

# 2. JOIN ENERGY + WEATHER
df_weather = df_weather.withColumnRenamed("dt_iso", "time")
df_clean = df_energy.join(df_weather, on="time", how="inner")

# 3. EKSTRAKSI FITUR WAKTU (Outlier IQR dipindah ke Modeling)
df_silver = df_clean \
    .withColumn("timestamp",   F.to_timestamp(F.col("time"))) \
    .withColumn("hour",        F.hour("timestamp")) \
    .withColumn("day_of_week", F.dayofweek("timestamp")) \
    .withColumn("month",       F.month("timestamp")) \
    .withColumn("year",        F.year("timestamp")) \
    .withColumn("is_weekend",  (F.dayofweek("timestamp").isin([1, 7])).cast("int"))

df_silver.write.mode("overwrite").parquet("/data/silver/energy_weather_clean")
print("\n=== Silver Layer Selesai (Tanpa Data Leakage) ✓ ===")
spark.stop()
```
Eksekusi:
```bash
docker exec -it tubes-k11-spark spark-submit /app/scripts/02_silver.py
cat ~/tubes_k11/data/silver/energy_weather_clean/*.snappy.parquet | mc pipe local/silver/energy_weather_clean/data.snappy.parquet
```


## 5. Gold Layer (One-Hot Encoding & Feature Importance)
```bash
nano ~/tubes_k11/scripts/03_gold.py
```
Isi dengan:
```bash
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import DecisionTreeRegressor

spark = SparkSession.builder.appName("tubes_k11_gold").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Tanpa dropna() global agar kolom yang 100% kosong tidak merusak dataset
df_silver = spark.read.parquet("/data/silver/energy_weather_clean")
df_silver.createOrReplaceTempView("energy_weather")

# 1. DESCRIPTIVE ANALYTICS
df_per_jam = spark.sql("SELECT hour, ROUND(AVG(`total load actual`), 2) AS avg_load_actual, ROUND(AVG(`total load forecast`), 2) AS avg_load_forecast FROM energy_weather GROUP BY hour ORDER BY hour")
df_per_jam.write.mode("overwrite").parquet("/data/gold/agg_per_jam")

df_renewable = spark.sql("SELECT year, ROUND(AVG(`generation solar`), 2) AS avg_solar, ROUND(AVG(`generation wind onshore`), 2) AS avg_wind_onshore, ROUND(AVG(`generation hydro run-of-river and poundage`), 2) AS avg_hydro, ROUND(AVG(`total load actual`), 2) AS avg_total_load FROM energy_weather GROUP BY year ORDER BY year")
df_renewable.write.mode("overwrite").parquet("/data/gold/agg_renewable")

# 2. FEATURE ENGINEERING (Ubah Kategori Kota jadi Angka)
numeric_features = ["temp", "pressure", "humidity", "wind_speed", "hour", "day_of_week", "is_weekend"]
target_col = "total load actual"

# HANYA dropna pada kolom yang akan dipakai untuk Machine Learning
df_ml = df_silver.select(numeric_features + ["city_name", target_col]).dropna()

indexer = StringIndexer(inputCol="city_name", outputCol="city_indexed")
encoder = OneHotEncoder(inputCols=["city_indexed"], outputCols=["city_encoded"])

df_indexed = indexer.fit(df_ml).transform(df_ml)
df_encoded = encoder.fit(df_indexed).transform(df_indexed)

all_features = numeric_features + ["city_encoded"]
assembler = VectorAssembler(inputCols=all_features, outputCol="features")
df_assembled = assembler.transform(df_encoded)
df_gold_model = df_assembled.select(F.col("features"), F.col(target_col).alias("label"))

# 3. FEATURE SELECTION (Decision Tree)
dt_eval = DecisionTreeRegressor(featuresCol="features", labelCol="label", maxDepth=5)
dt_model_eval = dt_eval.fit(df_gold_model)

importances = dt_model_eval.featureImportances.toArray()
imp_data = [(numeric_features[i], float(importances[i])) for i in range(len(numeric_features))]
imp_data.append(("city (encoded)", float(sum(importances[len(numeric_features):]))))

df_imp = spark.createDataFrame(imp_data, ["feature", "importance"])
df_imp.write.mode("overwrite").parquet("/data/gold/feature_importance")

df_gold_model.write.mode("overwrite").parquet("/data/gold/dataset_modeling")
print("\n=== Gold Layer Selesai ✓ ===")
spark.stop()
```
Eksekusi:
```bash
docker exec -it tubes-k11-spark spark-submit /app/scripts/03_gold.py
cat ~/tubes_k11/data/gold/dataset_modeling/*.snappy.parquet | mc pipe local/gold/dataset_modeling/data.snappy.parquet
```


## 6. Modeling (Decision Tree Regressor)
```bash
nano ~/tubes_k11/scripts/04_modeling.py
```
Isi dengan:
```bash
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 1. INISIALISASI SPARK DENGAN ALOKASI RAM EKSTRA (4GB)
spark = SparkSession.builder \
    .appName("tubes_K11") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Baca data bersih dari Silver Layer
df_silver = spark.read.parquet("/data/silver/energy_weather_clean")

numeric_features = ["temp", "pressure", "humidity", "wind_speed", "hour", "day_of_week", "is_weekend"]
target_col = "total load actual"

# Ambil kolom yang akan dipakai untuk ML & hapus baris kosong
df_ml = df_silver.select(numeric_features + ["city_name", F.col(target_col).alias("label")]).dropna()

print("\n=== 1. SPLIT DATA (MENCEGAH DATA LEAKAGE) ===")
df_train, df_test = df_ml.randomSplit([0.8, 0.2], seed=42)
df_train.cache()
df_test.cache()

print("\n=== 2. OUTLIER DETECTION (FIT HANYA PADA DATA TRAIN) ===")
features_to_check = ["temp", "pressure", "humidity", "wind_speed"]
bounds = {}

# Hitung IQR HANYA dari data Train
for col in features_to_check:
    quantiles = df_train.approxQuantile(col, [0.25, 0.75], 0.01)
    IQR = quantiles[1] - quantiles[0]
    bounds[col] = {
        'lower': quantiles[0] - 1.5 * IQR,
        'upper': quantiles[1] + 1.5 * IQR
    }

# Terapkan filter batas IQR ke Train dan Test
for col in features_to_check:
    df_train = df_train.filter((F.col(col) >= bounds[col]['lower']) & (F.col(col) <= bounds[col]['upper']))
    df_test = df_test.filter((F.col(col) >= bounds[col]['lower']) & (F.col(col) <= bounds[col]['upper']))

print("\n=== 3. ML PIPELINE (ENCODING & NORMALISASI) ===")
# Pipeline memastikan transformasi tidak bocor ke data Test
indexer = StringIndexer(inputCol="city_name", outputCol="city_indexed", handleInvalid="keep")
encoder = OneHotEncoder(inputCols=["city_indexed"], outputCols=["city_encoded"])

all_features = numeric_features + ["city_encoded"]
assembler = VectorAssembler(inputCols=all_features, outputCol="features_assembled")

scaler = StandardScaler(inputCol="features_assembled", outputCol="scaled_features", withStd=True, withMean=True)

# Bungkus dalam satu Pipeline dan Fit HANYA pada Train
pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler])
pipeline_model = pipeline.fit(df_train)

# Transformasi Train dan Test menjadi matriks siap pakai
df_train_scaled = pipeline_model.transform(df_train)
df_test_scaled = pipeline_model.transform(df_test)

print("\n=== 4. BIAS-VARIANCE TRADEOFF ===")
# Menggunakan kedalaman untuk mencegah Out Of Memory
depths = [2, 4, 6, 8, 10, 12]
train_mse, test_mse = [], []
evaluator_mse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mse")

for d in depths:
    # Set numTrees=20 (standar yang aman untuk memori)
    rf = RandomForestRegressor(featuresCol="scaled_features", labelCol="label", maxDepth=d, numTrees=20, seed=42)
    model = rf.fit(df_train_scaled)
    
    train_mse.append(evaluator_mse.evaluate(model.transform(df_train_scaled)))
    test_mse.append(evaluator_mse.evaluate(model.transform(df_test_scaled)))

# Plotting grafik Bias-Variance
os.makedirs("/data/visualisasi", exist_ok=True)
plt.figure(figsize=(8, 5))
plt.plot(depths, train_mse, marker='o', label='Train MSE (Loss)', color='blue')
plt.plot(depths, test_mse, marker='s', label='Validation/Test MSE (Loss)', color='orange')
plt.title('Bias-Variance Tradeoff: Random Forest')
plt.xlabel('Kompleksitas Model (Max Depth)')
plt.ylabel('Error (MSE)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("/data/visualisasi/06_bias_variance_tradeoff.png")
plt.close()

print("\n=== 5. FINAL EVALUATION ===")
# Dipilih maxDepth=10 yang sering menjadi titik optimal (Sweet Spot) sebelum Overfitting
final_rf = RandomForestRegressor(featuresCol="scaled_features", labelCol="label", maxDepth=10, numTrees=50, seed=42)
final_model = final_rf.fit(df_train_scaled)
final_predictions = final_model.transform(df_test_scaled)

# Evaluasi R-Squared
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
r2_score = evaluator_r2.evaluate(final_predictions)

# Evaluasi MAPE (Langkah manual untuk akurasi persentase)
df_mape = final_predictions.withColumn(
    "abs_pct_error", 
    F.abs((F.col("label") - F.col("prediction")) / F.col("label"))
)
mape_score = df_mape.agg(F.mean("abs_pct_error")).collect()[0][0] * 100

print(f"--- HASIL AKHIR RANDOM FOREST ---")
print(f"R-Squared (R²) : {r2_score:.4f} (Mendekati 1 semakin baik)")
print(f"MAPE           : {mape_score:.2f}% (Semakin kecil semakin baik)")

# Simpan prediksi untuk visualisasi akhir
final_predictions.select("label", "prediction").write.mode("overwrite").parquet("/data/gold/predictions_rf_final")
print("\n=== Modeling Selesai ✓ ===")
spark.stop()
```
Eksekusi:
```bash
docker exec -it tubes-k11-spark spark-submit /app/scripts/04_modeling.py
```



