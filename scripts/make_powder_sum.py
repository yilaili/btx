import logging
import numpy as np
import os

from sfx_utils.diagnostics.geom_opt import GeomOpt

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def task(config):
    geom_opt = GeomOpt(exp=config.exp,
                       run=config.run,
                       det_type=config.det_type)

    logger.debug(f'Computing Powder for run {config.run} of {config.exp}...')
    powder = geom_opt.compute_powder(n_images=config.n_images,
                                     batch_size=config.batch_size)

    powder_filepath=os.path.join(config.root_dir,f'run_{config.run}.npy')
    logger.info(f'Saving Powder to {powder_filepath}')
    np.save(powder_filepath, powder)
    logger.debug('Done!')