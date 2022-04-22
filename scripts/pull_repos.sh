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
echo "Moved to where the repositories are expected to be: $PWD"

repo_list_success=""
for repo in $repo_list; do
  if [ -d ../${repo} ]; then
    cd ../${repo}
    echo "> Updating $repo"
    echo "git pull origin main"
    git pull origin main
    repo_list_success=${repo_list_success}" $repo "
  else
    echo "Warning! $repo could not be updated."

curl -s -XPOST ${JID_UPDATE_COUNTERS} -H "Content-Type: application/json" -d '[ {"key": "<b>List of repository pulled</b>", "value": "'"${repo_list_success}"'" } ]'
