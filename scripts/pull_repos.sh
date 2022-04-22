#!/bin/bash
#
# We expect the following tree:
# root_dir/
#    L btx / scripts / < location of this script >
#    L mrxv
#    L omdevteam.github.io

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LOGFILE=${SCRIPT_DIR}/tmp.log

#Submit to SLURM
sbatch << EOF
#!/bin/bash

#SBATCH -p psanaq
#SBATCH -t 10:00:00
#SBATCH --exclusive
#SBATCH --job-name pull_repos
#SBATCH --ntasks=1

#repo_list='btx mrxv'
#echo "[SLURM]Attempting to update the following repositories: ${repo_list}" > ${LOGFILE}

cd $SCRIPT_DIR
cd ../
echo "[SLURM]Moved to where the repositories are expected to be: $PWD" >> ${LOGFILE}

if [ -d ../btx ]; then
  cd ../btx
  echo "[SLURM]> Updating btx" >> ${LOGFILE}
  git pull origin main 2>&1 ${LOGFILE}
else
  echo "[SLURM]Warning! btx could not be updated." >> ${LOGFILE}
fi

if [ -d ../mrxv ]; then
  cd ../mrxv
  echo "[SLURM]> Updating mrxv" >> ${LOGFILE}
  git pull origin main 2>&1 ${LOGFILE}
else
  echo "[SLURM]Warning! mrxv could not be updated." >> ${LOGFILE}
fi

#curl -s -XPOST ${JID_UPDATE_COUNTERS} -H "Content-Type: application/json" -d '[ {"key": "<b>List of repository pulled</b>", "value": "'"${repo_list_success}"'" } ]'
EOF

echo "Job sent to queue" >> ${LOGFILE}