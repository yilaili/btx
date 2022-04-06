import logging
import numpy as np
import os

from btx.diagnostics.run import RunDiagnostics
from btx.diagnostics.geom_opt import GeomOpt
from btx.misc.shortcuts import conditional_mkdir

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test(config):
    print(config)

def make_powder(config):
    """ Generate the max, avg, and std powders for a given run. """
    rd = RunDiagnostics(exp=config.exp,
                        run=config.run,
                        det_type=config.det_type)

    logger.debug(f'Computing Powder for run {config.run} of {config.exp}...')
    rd.compute_run_stats(n_images=config.n_images, 
                         powder_only=True)
    logger.info(f'Saving Powders to {config.root_dir}')
    rd.save_powders(config.root_dir)
    logger.debug('Done!')
    
def opt_distance(config):
    """ Optimize the detector distance from an AgBehenate run. """
    geom_opt = GeomOpt(exp=config.exp,
                       run=config.run,
                       det_type=config.det_type)
    config.center = tuple([float(elem) for elem in config.center.split()])
    logger.debug(f'Optimizing detector distance for run {config.run} of {config.exp}...')
    dist = geom_opt.opt_distance(powder=config.powder,
                                 center=config.center,
                                 plot=os.path.join(config.root_dir, config.plot))
    logger.info(f'Detector distance inferred from powder rings: {dist} mm')
    logger.debug('Done!')
