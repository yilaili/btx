import logging
import os
import requests

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Fetch the URL to post progress update
update_url = os.environ.get('JID_UPDATE_COUNTERS')

def test(config):
    print(config)
    requests.post(update_url, json=[config])

def make_powder(config):
    from btx.diagnostics.run import RunDiagnostics
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
    from btx.diagnostics.geom_opt import GeomOpt
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

def find_peaks(config):
    from btx.processing.peak_finder import PeakFinder
    setup = config.setup
    task = config.find_peaks
    """ Perform adaptive peak finding on run. """
    taskdir = os.path.join(setup.root_dir, 'index')
    os.makedirs(taskdir, exist_ok=True)
    pf = PeakFinder(exp=setup.exp, run=setup.run, det_type=setup.det_type, outdir=os.path.join(taskdir ,f"r{setup.run:04}"), 
                    tag=task.tag, mask=task.mask, psana_mask=task.psana_mask, min_peaks=task.min_peaks, max_peaks=task.max_peaks,
                    npix_min=task.npix_min, npix_max=task.npix_max, amax_thr=task.amax_thr, atot_thr=task.atot_thr, 
                    son_min=task.son_min, peak_rank=task.peak_rank, r0=task.r0, dr=task.dr, nsigm=task.nsigm)
    logger.debug(f'Performing peak finding for run {setup.run} of {setup.exp}...')
    pf.find_peaks()
    pf.curate_cxi()
    pf.summarize() 
    pf.report(update_url)
    logger.info(f'Saving CXI files and summary to {taskdir}/r{setup.run:04}')
    logger.debug('Done!')

def index(config):
    from btx.processing.indexer import Indexer
    setup = config.setup
    task = config.find_peaks
    """ Index run using indexamajig. """
    taskdir = os.path.join(setup.root_dir, 'index')
    indexer_obj = Indexer(exp=config.setup.exp, run=config.setup.run, det_type=config.setup.det_type, taskdir=taskdir, geom=config.index.geom,
                          cell=config.index.cell, int_rad=config.index.int_radius, methods=config.index.methods, tolerance=config.index.tolerance, 
                          no_revalidate=config.index.no_revalidate, multi=config.index.multi, profile=config.index.profile)
    logger.debug(f'Generating indexing executable for run {setup.run} of {setup.exp}...')
    indexer_obj.write_exe()
    logger.info(f'Executable written to {indexer_obj.tmp_exe}')
