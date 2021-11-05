import numpy as np
from psana_interface import *
from ag_behenate import *

class GeomOpt:
    
    def __init__(self, exp, run, det_type):
        self.psi = PsanaInterface(exp=exp, # experiment name, string
                                  run=run, # run number, int
                                  det_type=det_type) # detector name, string
        
    def _compute_powder(self, n_images=500, ptype='max'):
        """
        Compute the powder from the first n_images of the run, either by taking
        the maximum or average value of each pixel across the image series.
        
        Parameters
        ----------
        n_images : int
            number of diffraction images
        ptype : string
            if 'max', take the max pixel value
            if 'average', take the average pixel value
            
        Returns
        -------
        powder : numpy.ndarray, 2d
            powder diffraction image, in shape of assembled detector
        """
        if ptype == 'max':
            return np.amax(self.psi.get_images(n_images), axis=0)
        else:
            return np.mean(self.psi.get_images(n_images), axis=0)
    
    def opt_distance(self, sample='AgBehenate', n_images=500, plot=False):
        """
        Estimate the sample-detector distance based on the properties of the powder
        diffraction image. Currently only implemented for silver behenate.
        
        Parameters
        ----------
        sample : string
            sample type, e.g. 'AgBehenate'
        n_images : int
            number of diffraction images
        plot : bool
            if True, visualize results of distance estimation

        Returns
        -------
        distance : float
            estimated sample-detector distance in mm
        """
        powder = self._compute_powder(n_images)
        
        if sample == 'AgBehenate':
            ag_behenate = AgBehenate()
            distance = ag_behenate.opt_distance(powder,
                                                self.psi.estimate_distance(),
                                                self.psi.get_pixel_size(), 
                                                self.psi.get_wavelength(),
                                                plot=plot)
        else:
            print("Sorry, currently only implemented for silver behenate")
            return -1
