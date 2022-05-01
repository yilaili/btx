import logging
import os
import requests
import glob

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Fetch the URL to post progress update
update_url = os.environ.get('JID_UPDATE_COUNTERS')

def test(config):
    print(config)
    requests.post(update_url, json=[ { "key": "root_dir", "value": f"{config.setup.root_dir}" },
                                     { "key": "experiment_name", "value": "mfxp19619"} ])

def fetch_mask(config):
    from btx.interfaces.mask_interface import MaskInterface
    setup = config.setup
    task = config.fetch_mask
    """ Fetch most recent mask for this detector from mrxv. """
    taskdir = os.path.join(setup.root_dir, 'mask')
    os.makedirs(taskdir, exist_ok=True)
    mi = MaskInterface(exp=setup.exp,
                       run=setup.run,
                       det_type=setup.det_type)
    mi.retrieve_from_mrxv(mrxv_path=task.mrxv_path, dataset=task.dataset)
    logger.info(f'Saving Mask to {taskdir}')
    mi.save_mask(os.path.join(taskdir, f'r0000.npy'))
    logger.debug('Done!')

def run_analysis(config):
    from btx.diagnostics.run import RunDiagnostics
    setup = config.setup
    task = config.run_analysis
    """ Generate the max, avg, and std powders for a given run. """
    taskdir = os.path.join(setup.root_dir, 'powder')
    os.makedirs(taskdir, exist_ok=True)
    rd = RunDiagnostics(exp=setup.exp,
                        run=setup.run,
                        det_type=setup.det_type)
    logger.debug(f'Computing Powder for run {setup.run} of {setup.exp}...')
    rd.compute_run_stats(max_events=task.max_events, mask=task.mask)
    logger.info(f'Saving Powders and plots to {taskdir}')
    rd.save_powders(taskdir)
    rd.visualize_powder(output=os.path.join(taskdir, f"powder_r{rd.psi.run:04}.png"))
    rd.visualize_stats(output=os.path.join(taskdir, f"stats_r{rd.psi.run:04}.png"))
    logger.debug('Done!')
    
def opt_distance(config):
    from btx.diagnostics.geom_opt import GeomOpt
    from btx.misc.metrology import modify_crystfel_header, generate_geom_file
    setup = config.setup
    task = config.opt_distance
    """ Optimize the detector distance from an AgBehenate run. """
    taskdir = os.path.join(setup.root_dir, 'geom')
    os.makedirs(taskdir, exist_ok=True)
    geom_opt = GeomOpt(exp=setup.exp,
                       run=setup.run,
                       det_type=setup.det_type)
    task.center = tuple([float(elem) for elem in task.center.split()])
    logger.debug(f'Optimizing detector distance for run {setup.run} of {setup.exp}...')
    dist = geom_opt.opt_distance(powder=os.path.join(setup.root_dir, f"powder/powder_max_r{setup.run:04}.npy"),
                                 center=task.center,
                                 plot=os.path.join(taskdir, f'r{setup.run:04}.png'))
    logger.info(f'Detector distance inferred from powder rings: {dist} mm')
    temp_file = os.path.join(taskdir, 'temp.geom')
    geom_file = os.path.join(taskdir, f'r{setup.run:04}.geom')
    generate_geom_file(setup.exp, setup.run, setup.det_type, task.input_geom, temp_file, det_dist=dist)
    modify_crystfel_header(temp_file, geom_file)
    os.remove(temp_file)
    logger.info(f'CrystFEL geom file saved with updated coffset value to: {geom_file}')
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
    try:
        pf.report(update_url)
    except:
        logger.debug("Could not communicate with the elog update url")
    logger.info(f'Saving CXI files and summary to {taskdir}/r{setup.run:04}')
    logger.debug('Done!')

def index(config):
    from btx.processing.indexer import Indexer
    setup = config.setup
    task = config.index
    """ Index run using indexamajig. """
    taskdir = os.path.join(setup.root_dir, 'index')
    indexer_obj = Indexer(exp=config.setup.exp, run=config.setup.run, det_type=config.setup.det_type, taskdir=taskdir, geom=task.geom,
                          cell=task.cell, int_rad=task.int_radius, methods=task.methods, tolerance=task.tolerance, 
                          no_revalidate=task.no_revalidate, multi=task.multi, profile=task.profile)
    logger.debug(f'Generating indexing executable for run {setup.run} of {setup.exp}...')
    indexer_obj.write_exe()
    logger.info(f'Executable written to {indexer_obj.tmp_exe}')

def stream_analysis(config):
    from btx.interfaces.stream_interface import StreamInterface
    setup = config.setup
    task = config.stream_analysis
    """ Diagnostics including cell distribution and peakogram. """
    taskdir = os.path.join(setup.root_dir, 'index')
    stream_files = glob.glob(os.path.join(taskdir, f"r*{task.tag}.stream"))
    st = StreamInterface(input_files=stream_files, cell_only=False)
    if st.rank == 0:
        logger.debug(f'Read stream files: {stream_files}')
        st.plot_cell_parameters(output=os.path.join(taskdir, f"{task.tag}_cell_distribution.png"))
        st.plot_peakogram(output=os.path.join(taskdir, f"{task.tag}_peakogram.png"))
        logger.info(f'Peakogram and cell distribution generated for sample {task.tag}')
    logger.debug('Done!')
