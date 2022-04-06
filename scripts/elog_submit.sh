#!/bin/bash
# Adapted from Silke's script.
# TODO: ask for permission to use.

usage()
{
cat << EOF
$(basename "$0"):
  Script to launch python scripts from the eLog

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
#SBATCH --job-name "task"
#SBATCH --ntasks=$CORES

source /reg/g/psdm/etc/psconda.sh -py3
export PYTHONPATH="${PYTHONPATH}:/cds/sw/package/autosfx/btx"

echo "$MAIN_PY $CONFIGFILE"
$MAIN_PY -c $CONFIGFILE
EOF

echo "Job sent to queue"
