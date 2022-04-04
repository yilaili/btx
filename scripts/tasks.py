import logging
import numpy as np
import os

from btx.diagnostics.run import RunDiagnostics
from btx.misc.shortcuts import conditional_mkdir

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test(config):
    print(config)

def make_powder(config):
    rd = RunDiagnostics(exp=config.exp,
                        run=config.run,
                        det_type=config.det_type)

    logger.debug(f'Computing Powder for run {config.run} of {config.exp}...')
    rd.compute_run_stats(n_images=config.n_images, 
                         powder_only=True)
    logger.info(f'Saving Powders to {config.root_dir}')
    rd.save_powders(config.root_dir)
    logger.debug('Done!')
    
