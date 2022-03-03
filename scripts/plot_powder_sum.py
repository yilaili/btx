from sfx_utils.diagnostics.geom_opt import GeomOpt

def task(config):
    geom_opt = GeomOpt(exp=config.exp,
                       run=config.run,
                       det_type=config.det_type)

    powder = geom_opt.compute_powder(n_images=config.n_images,
                                     batch_size=config.batch_size,
                                     plot=config.do_plot)