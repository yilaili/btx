#!/bin/bash
#
# We expect the following tree:
# root_dir/
#    L btx / scripts / < location of this script >
#    L mrxv
#    L omdevteam.github.io

repo_list="btx mrxv omdevteam.github.io"
echo "Attempting to update the following repositories: ${repo_list}"

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

#repo_list_success=""
#for repo in ${repo_list}; do
#  if [ -d ../${repo} ]; then
#    cd ../${repo}
#    echo "> Updating ${repo}" >> ${SCRIPT_DIR}/tmp.log
#    echo "git pull origin main" >> ${SCRIPT_DIR}/tmp.log
#    git pull origin main
#    repo_list_success=${repo_list_success}" ${repo} "
#  else
#    echo "Warning! ${repo} could not be updated." >> ${SCRIPT_DIR}/tmp.log
#  fi
#done
#curl -s -XPOST ${JID_UPDATE_COUNTERS} -H "Content-Type: application/json" -d '[ {"key": "<b>List of repository pulled</b>", "value": "'"${repo_list_success}"'" } ]'
EOF

echo "Job sent to queue" >> ${LOGFILE}