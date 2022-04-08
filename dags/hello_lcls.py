from datetime import datetime
import uuid
from airflow import DAG
#from airflow.operators.python_operator import PythonOperator
from plugins.jid import JIDSlurmOperator

dag_id = "Hello_LCLS"

with DAG(dag_id=dag_id, start_date=datetime(2022, 3, 28),
         schedule_interval=None) as dag:

    def say_hello():
        print("Hello, LCLS! Your are the best!")

    #PythonOperator(task_id="say_hello", python_callable=say_hello)
    JIDSlurmOperator(
        task_id=str(uuid.uuid4()),
        slurm_script='/reg/g/psdm/tutorials/batchprocessing/arp_actual.py'
    )