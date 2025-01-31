from datetime import datetime
import os
from airflow import DAG
from plugins.jid import JIDSlurmOperator

# DAG SETUP
description='BTX stream analysis DAG'
dag_name = os.path.splitext(os.path.basename(__file__))[0]

dag = DAG(
    dag_name,
    start_date=datetime( 2022,4,1 ),
    schedule_interval=None,
    description=description,
  )


# Tasks SETUP

task_id='stream_analysis'
stream_analysis = JIDSlurmOperator( task_id=task_id, dag=dag, run_at='SRCF_FFB')

task_id='merge'
merge = JIDSlurmOperator( task_id=task_id, dag=dag, run_at='SRCF_FFB')

# Draw the DAG
stream_analysis >> merge