from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
}

with DAG(
    dag_id="mlops_training_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    train = BashOperator(
        task_id="run_training_pipeline",
        bash_command="cd /opt/airflow && python main.py",
    )

    train