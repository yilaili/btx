import numpy as np
import psana
from psana import DataSource

class PsanaInterface:

    def __init__(self, exp, run, det_type):
        self.exp = exp # experiment name, string
        self.run = run # run number, int
        self.det_type = det_type # detector name, string
        self.ds = psana.DataSource(f'exp={exp}:run={run}')
        self.det = psana.Detector(det_type, self.ds.env())
        
    def get_pixel_size(self):
        """
        Retrieve the detector's pixel size in millimeters.
        
        Returns
        -------
        pixel_size : float
            detector pixel size in mm
        """
        return self.det.pixel_size(self.ds.env()) / 1.0e3
    
    def get_wavelength(self):
        """
        Retrieve the detector's wavelength in Angstrom.
        
        Returns
        -------
        wavelength : float
            wavelength in Angstrom
        """
        return self.ds.env().epicsStore().value('SIOC:SYS0:ML00:AO192') * 10.
    
    def estimate_distance(self):
        """
        Retrieve an estimate of the detector distance in mm.
        
        Returns
        -------
        distance : float
            estimated detector distance
        """
        return -1*np.mean(self.det.coords_z(self.run))/1e3
    
    def get_images(self, num_images, assemble=True):
        """
        Retrieve a fixed number of images from the run.
        
        Paramters
        ---------
        num_images : int
            number of images to retrieve
        assemble : bool, default=True
            whether to assemble panels into image
            
        Returns
        -------
        images : numpy.ndarray, shape ((num_images,) + det_shape)
            images retrieved sequentially from run, optionally assembled
        """
        counter = 0
        if assemble:
            images = np.zeros((num_images, 
                               self.det.image_xaxis(self.run).shape[0], 
                               self.det.image_yaxis(self.run).shape[0]))
        else:
            images = np.zeros((num_images,) + self.det.shape())
        
        for num,evt in enumerate(self.ds.events()):
            if counter < num_images:
                if assemble:
                    images[counter] = self.det.image(evt=evt)
                else:
                    images[counter] = self.det.calib(evt=evt)
            else:
                break
            counter += 1
            
        return images
