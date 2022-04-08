import os
from datetime import datetime

from airflow import DAG
from airflow.models import BaseOperator, SkipMixin
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.plugins_manager import AirflowPlugin

from airflow.utils.decorators import apply_defaults

import getpass
import time
import requests

import logging
LOG = logging.getLogger(__name__)


class JIDBaseOperator( BaseOperator ):
  ui_color = '#006699'

  locations = {
    'SLAC': "http://psdm02:8446/jid_slac/",
    'SRCF_FFB': "http://psdm02:8446/jid_srcf_ffb/",
  }
  template_fields = ('experiment',)

  @apply_defaults
  def __init__(self,
      experiment: str,
      user=getpass.getuser(),
      run_at='SLAC',
      poke_interval=30,
      *args, **kwargs ):
    super( JIDBaseOperator, self ).__init__(*args, **kwargs)
    self.experiment = experiment
    self.user = user
    self.run_at = run_at
    self.poke_interval = poke_interval


class JIDSlurmOperator( BaseOperator ):

  ui_color = '#006699'

  locations = {
    'SLAC': "http://psdm02:8446/jid_slac/",
    'SRCF_FFB': "http://psdm02:8446/jid_srcf_ffb/",
  }

  endpoints = {
    'start_job': f'{experiment_name}/start_job',
    'job_statuses': '/job_statuses',
    'job_log_file': f'{experiment_name}/job_log_file',
  }

  template_fields = ['slurm_script','bash_commands',]

  @apply_defaults
  def __init__(self,
      slurm_script: str,
      bash_commands: str,
      user=getpass.getuser(),
      run_at='SLAC',
      poke_interval=30,
      working_dir='...', #TODO
      *args, **kwargs ):

    super(JIDSlurmOperator, self).__init__(*args, **kwargs)

    self.slurm_script = slurm_script
    self.bash_commands = bash_commands

    self.working_dir = working_dir
    self.user = user
    self.run_at = run_at
    self.poke_interval = poke_interval

  def create_control_doc( self, context, executable, parameters ):
    return {
      "_id" : executable,
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
    operators = [JIDBaseOperator,JIDSlurmOperator]