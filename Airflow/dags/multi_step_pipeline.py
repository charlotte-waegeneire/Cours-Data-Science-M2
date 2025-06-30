from airflow.decorators import dag, task
from datetime import datetime


@dag(
    dag_id="etl_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["etl"],
)
def etl_workflow():
    @task
    def extract():
        print("📥 Extracting data...")

    @task
    def transform():
        print("🔄 Transforming data...")

    @task
    def load():
        print("📤 Loading data...")

    extract() >> transform() >> load()


dag = etl_workflow()
