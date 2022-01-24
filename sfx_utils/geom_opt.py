import numpy as np
from psana_interface import *
from ag_behenate import *

class GeomOpt:
    
    def __init__(self, exp, run, det_type):
        self.psi = PsanaInterface(exp=exp, # experiment name, string
                                  run=run, # run number, int
                                  det_type=det_type) # detector name, string
        self.powder = None # for storing powder on the fly
        
    def compute_powder(self, n_images=500, batch_size=50, ptype='max'):
        """
        Compute the powder from the first n_images of the run, either by taking
        the maximum or average value of each pixel across the image series. The
        powder will be stored as a self variable and ovewrite any stored powder.
        
        Parameters
        ----------
        n_images : int
            total number of diffraction images to process
        batch_size: int
            number of images per batch
        ptype : string
            if 'max', take the max pixel value
            if 'average', take the average pixel value
            
        Returns
        -------
        powder : numpy.ndarray, 2d
            powder diffraction image, in shape of assembled detector
        """
        if batch_size > n_images:
            batch_size = n_images

        n_proc = 0
        while n_proc < n_images:
            try:
                images = self.psi.get_images(batch_size, assemble=True)
                
                if ptype == 'max':
                    if self.powder is None:
                        self.powder = np.max(images, axis=0)
                    else:
                        self.powder = np.max(np.concatenate((self.powder[np.newaxis,:,:],
                                                             images)), axis=0)
                elif ptype == 'mean':
                    if self.powder is None:
                        self.powder = np.sum(images, axis=0)
                    else:
                        self.powder = np.sum(np.concatenate((self.powder[np.newaxis,:,:],
                                                             images)), axis=0)
                else:
                    raise ValueError("Invalid powder type, must be max or mean")             
                n_proc += batch_size
            
            except:
                print("Reached end of run")
                break
                    
        if ptype == 'mean':
            self.powder /= float(n_proc)
        
        return self.powder

    def opt_distance(self, sample='AgBehenate', n_images=500, center=None, plot=False):
        """
        Estimate the sample-detector distance based on the properties of the powder
        diffraction image. Currently only implemented for silver behenate.
        
        Parameters
        ----------
        sample : string
            sample type, e.g. 'AgBehenate'
        n_images : int
            number of diffraction images
        center : tuple
            detector center (xc,yc) in pixels. if None, assume assembled image center.
        plot : bool
            if True, visualize results of distance estimation

        Returns
        -------
        distance : float
            estimated sample-detector distance in mm
        """
        if self.powder is None:
            self.powder = self.compute_powder(n_images)
        
        if sample == 'AgBehenate':
            ag_behenate = AgBehenate()
            distance = ag_behenate.opt_distance(self.powder,
                                                self.psi.estimate_distance(),
                                                self.psi.get_pixel_size(), 
                                                self.psi.get_wavelength(),
                                                center=center,
                                                plot=plot)
            return distance

        else:
            print("Sorry, currently only implemented for silver behenate")
            return -1
