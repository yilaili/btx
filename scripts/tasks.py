import logging
import os
import requests

from btx.diagnostics.run import RunDiagnostics
from btx.diagnostics.geom_opt import GeomOpt

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Fetch the URL to post progress update
update_url = os.environ.get('JID_UPDATE_COUNTERS')

def test(config):
    print(config)
    requests.post(update_url, json=[config])

def make_powder(config):
    setup = config.setup
    task = config.make_powder
    """ Generate the max, avg, and std powders for a given run. """
    rd = RunDiagnostics(exp=setup.exp,
                        run=setup.run,
                        det_type=setup.det_type)

    logger.debug(f'Computing Powder for run {setup.run} of {setup.exp}...')
    rd.compute_run_stats(n_images=task.n_images,
                         powder_only=True)
    logger.info(f'Saving Powders to {setup.root_dir}')
    rd.save_powders(setup.root_dir)
    logger.debug('Done!')
    
def opt_distance(config):
    setup = config.setup
    task = config.opt_distance
    """ Optimize the detector distance from an AgBehenate run. """
    geom_opt = GeomOpt(exp=setup.exp,
                       run=setup.run,
                       det_type=setup.det_type)
    task.center = tuple([float(elem) for elem in task.center.split()])
    logger.debug(f'Optimizing detector distance for run {setup.run} of {setup.exp}...')
    dist = geom_opt.opt_distance(powder=task.powder,
                                 center=task.center,
                                 plot=os.path.join(setup.root_dir, task.plot))
    logger.info(f'Detector distance inferred from powder rings: {dist} mm')
    logger.debug('Done!')
