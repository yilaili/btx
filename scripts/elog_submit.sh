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
    *)
      POSITIONAL+=("$1")
      shift
      ;;
    esac
done
set -- "${POSITIONAL[@]}"

QUEUE=${QUEUE:='psanaq'}
CORES=32
MAIN_PY='/cds/sw/package/autosfx/sfx_utils/main.py'

#Submit to SLURM
sbatch << EOF
#!/bin/bash

#SBATCH -p ${QUEUE}
#SBATCH -t 10:00:00
#SBATCH --exclusive
#SBATCH --job-name "task"
#SBATCH --ntasks=$CORES

source /reg/g/psdm/etc/psconda.sh -py3

echo "$MAIN_PY $CONFIGFILE"
$MAIN_PY $CONFIGFILE
EOF

echo "Job sent to queue"