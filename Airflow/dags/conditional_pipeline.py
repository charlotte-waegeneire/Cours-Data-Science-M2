from airflow.decorators import dag, task, task_group
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from datetime import datetime
import random


@dag(
    dag_id="branching_example",
    start_date=datetime(2023, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["branch"],
)
def branching_workflow():
    def choose_path():
        return "path_a" if random.random() < 0.5 else "path_b"

    start = BranchPythonOperator(task_id="choose_path", python_callable=choose_path)

    @task(task_id="path_a")
    def path_a():
        print("✅ Running path A")

    @task(task_id="path_b")
    def path_b():
        print("✅ Running path B")

    join = EmptyOperator(task_id="join", trigger_rule="none_failed_min_one_success")

    start >> [path_a(), path_b()] >> join


dag = branching_workflow()
