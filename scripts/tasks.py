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
    requests.post(update_url, json=[ { "key": "root_dir", "value": f"{config.setup.root_dir}" } ])

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
    mi.retrieve_from_mrxv(dataset=task.dataset)
    logger.info(f'Saving mrxv mask to {taskdir}')
    mi.save_mask(os.path.join(taskdir, f'r0000.npy'))
    logger.debug('Done!')

def fetch_geom(config):
    from btx.misc.metrology import retrieve_from_mrxv
    setup = config.setup
    task = config.fetch_geom
    """ Fetch latest geometry for this detector from mrxv. """
    taskdir = os.path.join(setup.root_dir, 'geom')
    os.makedirs(taskdir, exist_ok=True)
    logger.info(f'Saving mrxv geom to {taskdir}')
    retrieve_from_mrxv(det_type=setup.det_type, out_geom=os.path.join(taskdir, f'r0000.geom'))
    logger.debug('Done!')

def build_mask(config):
    from btx.interfaces.mask_interface import MaskInterface
    from btx.misc.shortcuts import fetch_latest
    setup = config.setup
    task = config.build_mask
    """ Generate a mask by thresholding events from a psana run. """
    taskdir = os.path.join(setup.root_dir, 'mask')
    os.makedirs(taskdir, exist_ok=True)
    mi = MaskInterface(exp=setup.exp,
                       run=setup.run,
                       det_type=setup.det_type)
    if task.combine:
        mask_file = fetch_latest(fnames=os.path.join(taskdir, 'r*.npy'), run=setup.run)
        mi.load_mask(mask_file, mask_format='psana')
        logger.debug(f'New mask will be combined with {mask_file}')
    task.thresholds = tuple([float(elem) for elem in task.thresholds.split()])
    mi.generate_from_psana_run(thresholds=task.thresholds, n_images=task.n_images, n_edge=task.n_edge)
    logger.info(f'Saving newly generated mask to {taskdir}')
    mi.save_mask(os.path.join(taskdir, f'r{mi.psi.run:04}.npy'))
    logger.debug('Done!')

def run_analysis(config):
    from btx.diagnostics.run import RunDiagnostics
    from btx.misc.shortcuts import fetch_latest
    setup = config.setup
    task = config.run_analysis
    """ Generate the max, avg, and std powders for a given run. """
    taskdir = os.path.join(setup.root_dir, 'powder')
    os.makedirs(taskdir, exist_ok=True)
    os.makedirs(os.path.join(taskdir, 'figs'), exist_ok=True)
    mask_file = fetch_latest(fnames=os.path.join(setup.root_dir, 'mask', 'r*.npy'), run=setup.run)
    logger.debug(f'Applying mask: {mask_file}...')
    rd = RunDiagnostics(exp=setup.exp,
                        run=setup.run,
                        det_type=setup.det_type)
    logger.debug(f'Computing Powder for run {setup.run} of {setup.exp}...')
    rd.compute_run_stats(max_events=task.max_events, mask=mask_file, threshold=task.get('mean_threshold'))
    logger.info(f'Saving Powders and plots to {taskdir}')
    rd.save_powders(taskdir)
    rd.visualize_powder(output=os.path.join(taskdir, f"figs/powder_r{rd.psi.run:04}.png"))
    rd.visualize_stats(output=os.path.join(taskdir, f"figs/stats_r{rd.psi.run:04}.png"))
    logger.debug('Done!')
    
def opt_geom(config):
    from btx.diagnostics.geom_opt import GeomOpt
    from btx.misc.shortcuts import fetch_latest
    setup = config.setup
    task = config.opt_geom
    """ Optimize and deploy the detector geometry from a silver behenate run. """
    taskdir = os.path.join(setup.root_dir, 'geom')
    os.makedirs(taskdir, exist_ok=True)
    os.makedirs(os.path.join(taskdir, 'figs'), exist_ok=True)
    geom_opt = GeomOpt(exp=setup.exp,
                       run=setup.run,
                       det_type=setup.det_type)
    if geom_opt.rank == 0:
        mask_file = fetch_latest(fnames=os.path.join(setup.root_dir, 'mask', 'r*.npy'), run=setup.run)
        logger.debug(f'Optimizing detector distance for run {setup.run} of {setup.exp}...')
        geom_opt.opt_geom(powder=os.path.join(setup.root_dir, f"powder/r{setup.run:04}_max.npy"),
                          mask=mask_file,
                          distance=task.get('distance'),
                          n_iterations=task.get('n_iterations'), 
                          n_peaks=task.get('n_peaks'), 
                          threshold=task.get('threshold'),
                          plot=os.path.join(taskdir, f'figs/r{setup.run:04}.png'))
        try:
            geom_opt.report(update_url)
        except:
            logger.debug("Could not communicate with the elog update url")
        logger.info(f'Detector distance in mm inferred from powder rings: {geom_opt.distance}')
        logger.info(f'Detector center in pixels inferred from powder rings: {geom_opt.center}')
        logger.info(f'Detector edge resolution in Angstroms: {geom_opt.edge_resolution}')    
        geom_opt.deploy_geometry(taskdir)
        logger.info(f'Updated geometry files saved to: {taskdir}')
        logger.debug('Done!')

def find_peaks(config):
    from btx.processing.peak_finder import PeakFinder
    from btx.misc.shortcuts import fetch_latest
    setup = config.setup
    task = config.find_peaks
    """ Perform adaptive peak finding on run. """
    taskdir = os.path.join(setup.root_dir, 'index')
    os.makedirs(taskdir, exist_ok=True)
    mask_file = fetch_latest(fnames=os.path.join(setup.root_dir, 'mask', 'r*.npy'), run=setup.run)
    pf = PeakFinder(exp=setup.exp, run=setup.run, det_type=setup.det_type, outdir=os.path.join(taskdir ,f"r{setup.run:04}"),
                    event_receiver=setup.get('event_receiver'), event_code=setup.get('event_code'), event_logic=setup.get('event_logic'),
                    tag=task.tag, mask=mask_file, psana_mask=task.psana_mask, min_peaks=task.min_peaks, max_peaks=task.max_peaks,
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
    from btx.misc.shortcuts import fetch_latest
    setup = config.setup
    task = config.index
    """ Index run using indexamajig. """
    taskdir = os.path.join(setup.root_dir, 'index')
    geom_file = fetch_latest(fnames=os.path.join(setup.root_dir, 'geom', 'r*.geom'), run=setup.run)
    indexer_obj = Indexer(exp=config.setup.exp, run=config.setup.run, det_type=config.setup.det_type, tag=task.tag, 
                          taskdir=taskdir, geom=geom_file, cell=task.get('cell'), int_rad=task.int_radius, methods=task.methods, 
                          tolerance=task.tolerance, tag_cxi=task.get('tag_cxi'), no_revalidate=task.no_revalidate, multi=task.multi, profile=task.profile)
    logger.debug(f'Generating indexing executable for run {setup.run} of {setup.exp}...')
    indexer_obj.write_exe()
    logger.info(f'Executable written to {indexer_obj.tmp_exe}')

def stream_analysis(config):
    from btx.interfaces.stream_interface import StreamInterface, write_cell_file
    setup = config.setup
    task = config.stream_analysis
    """ Diagnostics including cell distribution and peakogram. Concatenate streams. """
    taskdir = os.path.join(setup.root_dir, 'index')
    os.makedirs(os.path.join(taskdir, 'figs'), exist_ok=True)
    stream_files = os.path.join(taskdir, f"r*{task.tag}.stream")
    st = StreamInterface(input_files=glob.glob(stream_files), cell_only=False)
    if st.rank == 0:
        logger.debug(f'Read stream files: {stream_files}')
        st.plot_cell_parameters(output=os.path.join(taskdir, f"figs/cell_{task.tag}.png"))
        st.plot_peakogram(output=os.path.join(taskdir, f"figs/peakogram_{task.tag}.png"))
        logger.info(f'Peakogram and cell distribution generated for sample {task.tag}')
        celldir = os.path.join(setup.root_dir, 'cell')
        os.makedirs(celldir, exist_ok=True)
        write_cell_file(st.cell_params, os.path.join(celldir, f"{task.tag}.cell"), input_file=setup.get('cell'))
        logger.info(f'Input stream files: {stream_files}')
        logger.info(f'Concatenating all stream files to {task.tag}.stream')
        stream_cat = os.path.join(taskdir, f"{task.tag}.stream")
        os.system(f"cat {stream_files} >> {stream_cat}")
        logger.debug('Done!')

def merge(config):
    from btx.processing.merge import StreamtoMtz
    setup = config.setup
    task = config.merge
    """ Merge reflections from stream file and convert to mtz. """
    taskdir = os.path.join(setup.root_dir, 'merge')
    os.makedirs(taskdir, exist_ok=True)
    input_stream = os.path.join(setup.root_dir, f"index/{task.tag}.stream")
    cellfile = os.path.join(setup.root_dir, f"cell/{task.tag}.cell")
    foms = task.foms.split(" ")
    stream_to_mtz = StreamtoMtz(input_stream, task.symmetry, taskdir, cellfile)
    stream_to_mtz.cmd_partialator(iterations=task.iterations, model=task.model, min_res=task.get('min_res'), push_res=task.get('push_res'))
    for ns in [1, task.nshells]:
        stream_to_mtz.cmd_compare_hkl(foms=foms, nshells=ns, highres=task.get('highres'))
    stream_to_mtz.cmd_report(foms=foms, nshells=task.nshells)
    stream_to_mtz.cmd_get_hkl()
    logger.info(f'Executable written to {stream_to_mtz.tmp_exe}')
