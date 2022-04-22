#!/bin/bash
#
# We expect the following tree:
# root_dir/
#    L btx / scripts / < location of this script >
#    L mrxv
#    L omdevteam.github.io

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
cd ../

LOGFILE=${SCRIPT_DIR}/tmp.log
echo "Moved to where the repositories are expected to be: $PWD" > ${LOGFILE}

#Submit to SLURM
sbatch << EOF
#!/bin/bash

#SBATCH -p psanaq
#SBATCH -t 10:00:00
#SBATCH --exclusive
#SBATCH --job-name pull_repos
#SBATCH --ntasks=1

repo_list='btx mrxv'
echo "[SLURM]Attempting to update the following repositories: ${repo_list}" >> ${LOGFILE}

repo_list_success=""
for repo in ${repo_list}; do
  if [ -d ../${repo} ]; then
    cd ../${repo}
    echo "[SLURM]> Updating ${repo}" >> ${LOGFILE}
    echo "[SLURM]git pull origin main" >> ${LOGFILE}
    git pull origin main 2>&1 ${LOGFILE}
    repo_list_success=${repo_list_success}" ${repo} "
  else
    echo "[SLURM]Warning! ${repo} could not be updated." >> ${LOGFILE}
  fi
done
echo "[SLURM]List of repository pulled: ${repo_list_success}" >> ${LOGFILE}
#curl -s -XPOST ${JID_UPDATE_COUNTERS} -H "Content-Type: application/json" -d '[ {"key": "<b>List of repository pulled</b>", "value": "'"${repo_list_success}"'" } ]'
EOF

echo "Job sent to queue" >> ${LOGFILE}