from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

dag_id = "Hello_LCLS"

with DAG(dag_id=dag_id, start_date=datetime(2022, 3, 28),
         schedule_interval=None) as dag:

    def say_hello():
        print("Hello, LCLS! Your are the best!")

    PythonOperator(task_id="say_hello", python_callable=say_hello)