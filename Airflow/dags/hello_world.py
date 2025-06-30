from airflow.decorators import dag, task
from datetime import datetime


@dag(
    dag_id="hello_world",
    start_date=datetime(2023, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["demo"],
)
def hello_workflow():
    @task
    def say_hello():
        print("👋 Hello from Airflow!")

    say_hello()


dag = hello_workflow()
