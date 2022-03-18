#!/usr/bin/env python

import argparse
import shutil
import yaml

from sfx_utils.misc.shortcuts import AttrDict, conditional_mkdir
from .tasks import make_powder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to config file.')
    config_filepath = parser.parse_args().config
    with open(config_filepath, "r") as config_file:
        config = AttrDict(yaml.safe_load(config_file))

    if not conditional_mkdir(config.root_dir):
        print(f"Error: cannot create root path.")
        return -1

    shutil.copy2(config_filepath, config.root_dir)

    if(config.task == 'make_powder'):
        make_powder(config)

    return 0, 'Task successfully executed'

if __name__ == '__main__':
    try:
        retval, status_message = main()
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = 'Error: Training failed.'

    print(status_message)
    exit(retval)