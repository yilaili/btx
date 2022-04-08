from datetime import datetime
import os
from airflow import DAG
from plugins.jid import JIDSlurmOperator

# TEMPORARY DEFINITIONS
slurm_script_directory="/cds/sw/package/autosfx/btx/adhoc/"

# DAG SETUP
description='BTX detector distance estimation DAG'
dag_name = os.path.splitext(os.path.basename(__file__))[0]

default_args = {
        'start_date': datetime( 2020,1,1 ),
}

dag = DAG(
    dag_name,
    default_args=default_args,
    description=description,
  )


# Tasks SETUP
task_id='make_powder'
make_powder = JIDSlurmOperator( task_id=task_id,
                                slurm_script=f'{slurm_script_directory}{task_id}.slurm'
                                dag=dag)

task_id='opt_distance'
opt_distance = JIDSlurmOperator( task_id=task_id,
                                 slurm_script=f'{slurm_script_directory}{task_id}.slurm',
                                 dag=dag)

# Draw the DAG
make_powder >> opt_distance
