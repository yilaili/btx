import os
from datetime import datetime

from airflow import DAG
from airflow.models import BaseOperator, SkipMixin
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.plugins_manager import AirflowPlugin

from airflow.utils.decorators import apply_defaults

import time
import requests

import logging
LOG = logging.getLogger(__name__)


class JIDBaseOperator( BaseOperator ):
  ui_color = '#006699'

  locations = {
    'SLAC': "https://pslogin01:8443/jid_slac/jid/ws",
  }
  template_fields = ('experiment',)

  @apply_defaults
  def __init__(self,
      experiment: str,
      user='mshankar',
      run_at='SLAC',
      poke_interval=30,
      cert='/usr/local/airflow/dags/certs/airflow.crt', key='/usr/local/airflow/dags/certs/airflow.key', root_ca='/usr/local/airflow/dags/certs/rootCA.crt', xcom_push=True,
      *args, **kwargs ):
    super( JIDBaseOperator, self ).__init__(*args, **kwargs)
    self.experiment = experiment
    self.user = user
    self.run_at = run_at
    self.poke_interval = poke_interval
    self.requests_opts = {
      "cert": ( cert, key ),
      "verify": False,
    }


class JIDJobOperator( JIDBaseOperator ):

  ui_color = '#006699'
  template_fields = ['experiment', 'run_id', 'executable', 'parameters' ]

  endpoints = {
    'start_job': '/start_job',
    'job_statuses': '/job_statuses',
    'job_log_file': '/job_log_file',
  }

  @apply_defaults
  def __init__(self,
      run_id: str,
      executable: str, parameters: str,
      *args, **kwargs ):
    super(JIDJobOperator, self).__init__(*args, **kwargs)
    self.workflow_id = self.experiment
    self.run_id = run_id
    self.executable = executable
    self.parameters = parameters

  def create_control_doc( self, context ):
    def_id = f"{self.experiment}-{self.run_id}-{context['task'].task_id}"
    return {
      "_id" : def_id,
      "experiment": self.experiment,
      "run_num" : self.run_id,
      "user" : self.user,
      "status" : '',
      "tool_id" : '', # lsurm job id
      "def_id" : def_id,
      "def": {
        "_id" : def_id,
        "name" : context['task'].task_id,
        "executable" : str(self.executable),
        "trigger" : "MANUAL",
        "location" : self.run_at,
        "parameters" : self.parameters,
        "run_as_user" : self.user
      }
    }


  def rpc( self, endpoint, control_doc, check_for_error=[] ):

    if not self.run_at in self.locations:
      raise AirflowException(f"JID location {self.run_at} is not configured")

    uri = self.locations[self.run_at] + self.endpoints[endpoint]
    LOG.info( f"Calling {uri} with {control_doc}..." )
    resp = requests.post( uri, json=control_doc, **self.requests_opts )
    LOG.info(f" + {resp.status_code}: {resp.content.decode('utf-8')}")
    if not resp.status_code in ( 200, ):
      raise AirflowException(f"Bad response from JID {resp}: {resp.content}")
    try:
      j = resp.json()
      if not j.get('success',"") in ( True, ):
        raise AirflowException(f"Error from JID {resp}: {resp.content}")
      v = j.get('value')
      # throw error if the response has any matching text: TODO use regex
      for i in check_for_error:
        if i in v:
          raise AirflowException(f"Response failed due to string match {i} against response {v}")
      return v
    except Exception as e:
      raise AirflowException(f"Response from JID not parseable: {e}")

  def execute(self, context):

    LOG.info(f"Attempting to run at {self.run_at}...")
    control_doc = self.create_control_doc( context )

    msg = self.rpc( 'start_job', control_doc )
    LOG.info(f"GOT job {msg}")
    jobs = [ msg, ]
    # pool and wait
    # TODO: ensure we don't do this forever
    time.sleep(10)
    while jobs[0].get('status') in ('RUNNING', 'SUBMITTED'):
      jobs = self.rpc( 'job_statuses', jobs, check_for_error=( ' error: ', 'Traceback ' ) )
      LOG.info("  waiting...")
      time.sleep(self.poke_interval)

    # grab logs and put into xcom
    out = self.rpc( 'job_log_file', jobs[0] )
    context['task_instance'].xcom_push(key='log',value=out)








class JIDFileBase( JIDBaseOperator ):
  ui_color = '#b19cd9'
  endpoints = {
    'file': '/file',
  }

  @apply_defaults
  def __init__(self,
    filepath=None,
    *args, **kwargs):
    super( JIDFileBase,  self ).__init__(*args, **kwargs)
    self.filepath = filepath

  def get_uri( self ):
    if not self.run_at in self.locations:
      raise AirflowException(f"JID location {self.run_at} is not configured")
    uri = self.locations[self.run_at] + self.endpoints['file'] + self.filepath
    return uri

  def parse( self, resp ):
    LOG.info(f"  {resp.status_code}: {resp.content}")
    if not resp.status_code in ( 200, ):
      raise AirflowException(f"Bad response from JID {resp}: {resp.content}")
    try:
      j = resp.json()
      if not j.get('success',"") in ( True, ):
        raise AirflowException(f"Error from JID {resp}: {resp.content}")
      return j.get('value')
    except Exception as e:
      raise AirflowException(f"Response from JID not parseable: {e}")

class LsSensor( JIDFileBase ):

  @apply_defaults
  def __init__(self,
    directory,
    *args, **kwargs):
    super( LsSensor,  self ).__init__(*args, **kwargs)
    self.directory = directory
    # full paths?

  def get_uri( self ):
    if not self.run_at in self.locations:
      raise AirflowException(f"JID location {self.run_at} is not configured")
    uri = self.locations[self.run_at] + self.endpoints['file'] + self.directory
    return uri

  def execute( self, context ):
    uri = self.get_uri()
    LOG.info( f"Calling {uri}..." )
    resp = requests.get( uri, **self.requests_opts )
    v = self.parse( resp )
    if not 'entries' in v:
      raise AirflowException(f"Error, could not find 'entries' in response")
    context['ti'].xcom_push( key='return_value', value=v['entries'] )


class GetFileSensor( JIDFileBase ):

  def execute( self, context ):
    uri = self.get_uri()
    LOG.info( f"Calling {uri}..." )
    resp = requests.get( uri, **self.requests_opts )
    #LOG.info(f"  {resp.status_code}: {resp.content}")
    if not resp.status_code in ( 200, ):
      raise AirflowException(f"Bad response from JID {resp}: {resp.content}")
    context['ti'].xcom_push( key='return_value', value=resp.content )

class PutFileOperator( JIDFileBase ):

  @apply_defaults
  def __init__(self,
    filepath,
    data,
    *args, **kwargs):
    super( PutFileOperator,  self ).__init__(*args, **kwargs)
    self.filepath = filepath
    self.data = data

  def execute( self, context ):
    uri = self.get_uri()
    LOG.info( f"Calling {uri}..." )
    resp = requests.put( uri, data=self.data, **self.requests_opts )
    v = self.parse( resp )
    if 'status' in v and v['status'] == 'ok':
      return True
    return False

class JIDSlurmOperator( BaseOperator ):

  ui_color = '#006699'

  locations = {
    'SLAC': "https://pslogin01:8443/jid_slac/jid/ws",
  }
  endpoints = {
    'start_job': '/start_job',
    'job_statuses': '/job_statuses',
    'job_log_file': '/job_log_file',
    'file': '/file',
  }

  template_fields = ['slurm_script','bash_commands',]

  @apply_defaults
  def __init__(self,
      slurm_script: str,
      bash_commands: str,
      user='mshankar',
      run_at='SLAC',
      poke_interval=30,
      working_dir='/global/project/projectdirs/lcls/autosfx/jobs',
      cert='/usr/local/airflow/dags/certs/airflow.crt', key='/usr/local/airflow/dags/certs/airflow.key', root_ca='/usr/local/airflow/dags/certs/rootCA.crt', xcom_push=True,
      *args, **kwargs ):

    super(JIDSlurmOperator, self).__init__(*args, **kwargs)

    self.slurm_script = slurm_script
    self.bash_commands = bash_commands

    self.working_dir = working_dir
    self.user = user
    self.run_at = run_at
    self.poke_interval = poke_interval
    self.requests_opts = {
      "cert": ( cert, key ),
      "verify": False,
    }

  def create_control_doc( self, context, executable, parameters ):
    return {
      "_id" : executable,
      #"experiment": context.get('dag_run').conf.get('experiment_name'),
      #"run_num" : context.get('dag_run').conf.get('run_num'),
      "experiment": context.get('dag_run').conf.get('experiment'),
      "run_num" : context.get('dag_run').conf.get('run_id'),
      "user" : self.user,
      "status" : '',
      "tool_id" : '', # lsurm job id
      "def_id" : executable,
      "def": {
        "_id" : executable,
        "name" : context['task'].task_id,
        "executable" : f"{self.working_dir}/{executable}.slurm",
        "trigger" : "MANUAL",
        "location" : self.run_at,
        "parameters" : parameters,
        "run_as_user" : self.user
      }
    }

  def get_file_uri( self, filepath ):
    if not self.run_at in self.locations:
      raise AirflowException(f"JID location {self.run_at} is not configured")
    uri = self.locations[self.run_at] + self.endpoints['file'] + filepath
    return uri

  def parse( self, resp ):
    LOG.info(f"  {resp.status_code}: {resp.content}")
    if not resp.status_code in ( 200, ):
      raise AirflowException(f"Bad response from JID {resp}: {resp.content}")
    try:
      j = resp.json()
      if not j.get('success',"") in ( True, ):
        raise AirflowException(f"Error from JID {resp}: {resp.content}")
      return j.get('value')
    except Exception as e:
      raise AirflowException(f"Response from JID not parseable: {e}")

  def put_file( self, path, content ):
    uri = self.get_file_uri( path )
    LOG.info( f"Calling {uri}..." )
    resp = requests.put( uri, data=content, **self.requests_opts )
    v = self.parse( resp )
    if 'status' in v and v['status'] == 'ok':
      return True
    return False

  def rpc( self, endpoint, control_doc, check_for_error=[] ):

    if not self.run_at in self.locations:
      raise AirflowException(f"JID location {self.run_at} is not configured")

    uri = self.locations[self.run_at] + self.endpoints[endpoint]
    LOG.info( f"Calling {uri} with {control_doc}..." )
    resp = requests.post( uri, json=control_doc, **self.requests_opts )
    LOG.info(f" + {resp.status_code}: {resp.content.decode('utf-8')}")
    if not resp.status_code in ( 200, ):
      raise AirflowException(f"Bad response from JID {resp}: {resp.content}")
    try:
      j = resp.json()
      if not j.get('success',"") in ( True, ):
        raise AirflowException(f"Error from JID {resp}: {resp.content}")
      v = j.get('value')
      # throw error if the response has any matching text: TODO use regex
      for i in check_for_error:
        if i in v:
          raise AirflowException(f"Response failed due to string match {i} against response {v}")
      return v
    except Exception as e:
      raise AirflowException(f"Response from JID not parseable: {e}")

  def execute( self, context ):

    LOG.info(f"Attempting to run at {self.run_at}...")

    #this = f"{context.get('dag_run').conf.get('experiment_name')}-{context.get('dag_run').conf.get('run_num')}-{context.get('task').task_id}"
    this = f"{context.get('dag_run').conf.get('experiment')}-{context.get('dag_run').conf.get('run_id')}-{context.get('task').task_id}"

    # upload slurm script it to the destination
    LOG.info("Uploading job scripts...")
    job_file = f"{self.working_dir}/{this}.slurm"
    self.put_file( job_file, self.slurm_script )

    command_file = f"{self.working_dir}/{this}.sh"
    self.put_file( command_file, self.bash_commands )

    # run job for it
    LOG.info("Queueing slurm job...")
    control_doc = self.create_control_doc( context, this, '' )
    msg = self.rpc( 'start_job', control_doc )
    LOG.info(f"jobid {msg['tool_id']} successfully submitted!")
    jobs = [ msg, ]

    # FIXME: initial wait for job to queue
    time.sleep(10)
    LOG.info("Checking for job completion...")
    while jobs[0].get('status') in ('RUNNING', 'SUBMITTED'):
      jobs = self.rpc( 'job_statuses', jobs, check_for_error=( ' error: ', 'Traceback ' ) )
      time.sleep(self.poke_interval)

    # grab logs and put into xcom
    out = self.rpc( 'job_log_file', jobs[0] )
    context['task_instance'].xcom_push(key='log',value=out)






class JIDPlugins(AirflowPlugin):
    name = 'jid_plugins'
    operators = [LsSensor,GetFileSensor,PutFileOperator,JIDBaseOperator,JIDJobOperator,JIDSlurmOperator]