#!/bin/bash

usage()
{
cat << EOF
$(basename "$0"):
  Script to launch python scripts from the eLog

  OPTIONS:
    -h|--help
      Definition of options
    -q|--queue
      Queue to use on SLURM
    -n|--ncores
      Number of cores
    -c|--config_file
      Input config file
    -t|--task
      Task name
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
    -q|--queue)
      QUEUE="$2"
      shift
      shift
      ;;
    -n|--ncores)
      CORES="$2"
      shift
      shift
      ;;
    -c|--config_file)
      CONFIGFILE="$2"
      shift
      shift
      ;;
    -t|--task)
      TASK="$2"
      shift
      shift
      ;;
    -j|--jid_update_counters)
      JID_UPDATE_COUNTERS_VAR="$2"
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

QUEUE=${QUEUE:='psanaq'}
CORES=${CORES:=1}
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MAIN_PY="${SCRIPT_DIR}/main.py"

#Submit to SLURM
sbatch << EOF
#!/bin/bash

#SBATCH -p ${QUEUE}
#SBATCH -t 10:00:00
#SBATCH --exclusive
#SBATCH --job-name ${TASK}
#SBATCH --ntasks=${CORES}

source /reg/g/psdm/etc/psconda.sh -py3  #TODO: get rid of hard-code
export PYTHONPATH="${PYTHONPATH}:$( dirname -- $SCRIPT_DIR})"

echo "$MAIN_PY -c $CONFIGFILE -t $TASK -j $JID_UPDATE_COUNTERS_VAR"
$MAIN_PY -c $CONFIGFILE -t $TASK -j $JID_UPDATE_COUNTERS_VAR
EOF

echo "Job sent to queue"
