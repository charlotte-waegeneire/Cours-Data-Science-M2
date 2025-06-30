from airflow.decorators import dag, task
from datetime import datetime
import pandas as pd
import requests
import os

url = "https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2023-01.csv"
csv_path = "/opt/airflow/data/taxi.csv"
cleaned_path = "/opt/airflow/data/taxi_cleaned.csv"


@dag(
    dag_id="real_world_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=["project"],
)
def real_pipeline():
    @task
    def download():
        os.makedirs("/opt/airflow/data", exist_ok=True)
        r = requests.get(url)
        with open(csv_path, "wb") as f:
            f.write(r.content)

    @task
    def clean():
        df = pd.read_csv(csv_path)
        df_clean = df.dropna()
        df_clean.to_csv(cleaned_path, index=False)

    download() >> clean()


dag = real_pipeline()
