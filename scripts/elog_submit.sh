#!/bin/bash

usage()
{
cat << EOF
$(basename "$0"):
  Script to launch python scripts from the eLog

  OPTIONS:
    -h|--help
      Definition of options
    -f|--facility
      Facility where we are running at
    -q|--queue
      Queue to use on SLURM
    -n|--ncores
      Number of cores
    -c|--config_file
      Input config file
    -e|--experiment_name
      Experiment Name
    -r|--run_number
      Run Number
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
    -f|--facility)
      FACILITY="$2"
      shift
      shift
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
    -e|--experiment_name)
      EXPERIMENT=$2
      shift
      shift
      ;;
    -r|--run_number)
      RUN_NUM=$2
      shift
      shift
      ;;
    -t|--task)
      TASK="$2"
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

FACILITY=${FACILITY:='SRCF_FFB'}
case $FACILITY in
  'SLAC')
    SIT_PSDM_DATA_DIR='/cds/data/psdm/'
    ;;
  'SRCF_FFB')
    SIT_PSDM_DATA_DIR='/cds/data/drpsrcf/'
    ;;
  *)
    echo "ERROR! $FACILITY is not recognized."
    ;;
esac

QUEUE=${QUEUE:='ffbh3q'}
CORES=${CORES:=1}
# TODO: find_peaks needs to be handled from ischeduler. For now we do this...
if [ ${TASK} != 'find_peaks' ]; then
  CORES=1
fi

EXPERIMENT=${EXPERIMENT:='None'}
RUN_NUM=${RUN_NUM:='None'}
THIS_CONFIGFILE=${CONFIGFILE}
if [ ${RUN_NUM} != 'None' ]; then
  THIS_CONFIGFILE="${CONFIGFILE%.*}_${RUN_NUM}.yaml"
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MAIN_PY="${SCRIPT_DIR}/main.py"
if [ ${CORES} -gt 1 ]; then
MAIN_PY="/cds/sw/ds/ana/conda1/inst/envs/ana-4.0.38-py3/bin/mpirun ${MAIN_PY}"
else
MAIN_PY="/cds/sw/ds/ana/conda1/inst/envs/ana-4.0.38-py3/bin/python ${MAIN_PY}"
fi

UUID=$(cat /proc/sys/kernel/random/uuid)
if [ "${HOME}" == '' ]; then
  TMP_DIR="${SCRIPT_DIR}"
else
  TMP_DIR="${HOME}/.btx/"
fi
mkdir -p $TMP_DIR
TMP_EXE="${TMP_DIR}/task_${UUID}.sh"

#Submit to SLURM
sbatch << EOF
#!/bin/bash

#SBATCH -p ${QUEUE}
#SBATCH -t 10:00:00
#SBATCH --exclusive
#SBATCH --job-name ${TASK}
#SBATCH --ntasks=${CORES}

source /reg/g/psdm/etc/psconda.sh -py3  #TODO: get rid of hard-code
conda env list | grep '*'
which mpirun
which python
export SIT_PSDM_DATA=${SIT_PSDM_DATA_DIR}
export PATH=/cds/sw/package/crystfel/crystfel-dev/bin:$PATH
export PYTHONPATH="${PYTHONPATH}:$( dirname -- ${SCRIPT_DIR})"
export NCORES=${CORES}
export TMP_EXE=${TMP_EXE}
export WHICHPYTHON='/cds/sw/ds/ana/conda1/inst/envs/ana-4.0.38-py3/bin/python'

if [ ${RUN_NUM} != 'None' ]; then
  echo "new config file: ${THIS_CONFIGFILE}"
  sed "s/run:/run: ${RUN_NUM} #/g" ${CONFIGFILE} > ${THIS_CONFIGFILE}
fi

echo "$MAIN_PY -c ${THIS_CONFIGFILE} -t $TASK"
$MAIN_PY -c ${THIS_CONFIGFILE} -t $TASK
if [ ${RUN_NUM} != 'None' ]; then
  rm -f ${THIS_CONFIGFILE}
fi
EOF

echo "Job sent to queue"
