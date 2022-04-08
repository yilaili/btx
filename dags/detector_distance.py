from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from plugins.jid import JIDSlurmOperator

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
make_powder = JIDSlurmOperator( task_id='make_powder')

opt_distance = JIDSlurmOperator( task_id='opt_distance')

# Draw the DAG
make_powder >> opt_distance
