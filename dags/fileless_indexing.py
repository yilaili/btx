from datetime import datetime
import os
from airflow import DAG
from plugins.jid import JIDSlurmOperator, LsSensor

# DAG SETUP
description='BTX detector distance estimation DAG'
dag_name = os.path.splitext(os.path.basename(__file__))[0]

dag = DAG(
    dag_name,
    start_date=datetime( 2022,4,1 ),
    schedule_interval=None,
    description=description,
  )

# Tasks SETUP
task_id='launch_cheetah'
launch_cheetah = JIDSlurmOperator( task_id=task_id, dag=dag,
                                   slurm_script=f'{task_id}.sh')

is_cheetah_running = LsSensor( task_id=task_id, dag=dag)

task_id='launch_crystfel'
launch_crystfel = JIDSlurmOperator( task_id=task_id, dag=dag,
                                    slurm_script=f'{task_id}.sh')

# Draw the DAG
launch_cheetah
is_cheetah_running >> launch_crystfel