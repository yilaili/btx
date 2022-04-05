#!/bin/bash

usage()
{
cat << EOF
$(basename "$0"):
  Script to trigger Airflow DAG run from the eLog

  OPTIONS:
    -h|--help
      Definition of options
    -c|--config
      Input config file
    -q|--queue
      Queue to use on SLURM
    -n|--ncores
      Number of cores
EOF
}

POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"

  case $key in
    -h|--help)
      usage
      exit
      ;;
    -d|--dag)
      DAG="$2"
      shift
      shift
      ;;
    -q|--queue)
      QUEUE="$2"
      shift
      shift
      ;;
    -c|--config)
      CONFIGFILE="$2"
      shift
      shift
      ;;
    -n|--ncores)
      CORES="$2"
      shift
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
    esac
done
set -- "${POSITIONAL[@]}"

DAG=${DAG:='Hello_LCLS'}
QUEUE=${QUEUE:='psanaq'}
CORES=${CORES:=1}


### below is inherited from previous Airflow, needs to be fixed ###

NETRC=/reg/neh/home5/fpoitevi/netrc
AIRFLOW_URL=pswww.slac.stanford.edu/airflow-dev/home

# test to make sure its up
curl --netrc-file ${NETRC}  https://${AIRFLOW_URL}/api/experimental/test

# temporary re-routing:
JID_UPDATE_COUNTERS=${JID_UPDATE_COUNTERS///psdm02.pcdsn/pslogin01.slac.stanford.edu}

# create new dagrun based on new RUNID
### prepare data to be passed on to curl
run_id="${EXPERIMENT}-${RUN_NUM}"

config="{"
config="${config}\\\"experiment\\\":\\\"${EXPERIMENT}\\\""
config="${config},"
config="${config}\\\"run_id\\\":\\\"${RUN_NUM}\\\""
config="${config},"
config="${config}\\\"JID_UPDATE_COUNTERS\\\":\\\"${JID_UPDATE_COUNTERS}\\\""
config="${config},"
config="${config}\\\"config_file\\\":\\\"${CONFIGFILE}\\\""
config="${config}}"

curl_data="{"
curl_data="${curl_data}\"run_id\":\"$run_id\""
curl_data="${curl_data},"
curl_data="${curl_data}\"conf\":\"$config\""
curl_data="${curl_data}}"
### done preparing data to be passed on to curl

curl --netrc-file ${NETRC} -X POST \
  https://${AIRFLOW_URL}/api/experimental/dags/${DAG}/dag_runs \
  -H 'Cache-Control: no-cache' \
  -H 'Content-Type: application/json' \
  -d $curl_data
