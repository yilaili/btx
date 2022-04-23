#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR
cd ../

echo "User: `whoami` | Location: $PWD"

for repo in "btx mrxv omdevteam.github.io"; do
  repo_path=../${repo}
  if [ -d $repo_path ]; then
    cd $repo_path
    git pull origin main
  else
    echo "Warning! ${repo} could not be updated."
  fi
done