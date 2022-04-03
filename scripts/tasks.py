import logging
import numpy as np
import os

from sfx_utils.diagnostics.run import RunDiagnostics
from sfx_utils.misc.shortcuts import conditional_mkdir

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test(config):
    print(config)

def make_powder(config):
    out_dir=os.path.join(config.root_dir, f'run{config.run}')
    if not conditional_mkdir(out_dir):
        print(f"Error: cannot create run path.")
        return -1

    rd = RunDiagnostics(exp=config.exp,
                        run=config.run,
                        det_type=config.det_type)

    logger.debug(f'Computing Powder for run {config.run} of {config.exp}...')
    rd.compute_run_stats(n_images=config.n_images, 
                         powder_only=True)
    logger.info(f'Saving Powders to {out_dir}')
    rd.save_powders(out_dir)
    logger.debug('Done!')
    
