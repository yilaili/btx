import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from sfx_utils.interfaces.psana_interface import PsanaInterface
from .ag_behenate import *

class GeomOpt:
    
    def __init__(self, exp, run, det_type):
        self.psi = PsanaInterface(exp=exp, # experiment name, string
                                  run=run, # run number, int
                                  det_type=det_type) # detector name, string
        self.powder = None # for storing powder on the fly
        
    def compute_powder(self, n_images=500, batch_size=50, ptype='max', plot=False):
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
        plot: bool
            if True, displays the powder image and histogram.
            
        Returns
        -------
        powder : numpy.ndarray, 2d
            powder diffraction image, in shape of assembled detector
        """
        if batch_size > n_images:
            batch_size = n_images

        n_proc = 0
        while n_proc < n_images:
            
            images = self.psi.get_images(batch_size, assemble=True)
    
            if ptype == 'max':
                if self.powder is None:
                    self.powder = np.max(images, axis=0)
                else:
                    self.powder = np.max(np.concatenate((self.powder[np.newaxis,:,:], images)), axis=0)
            
            elif ptype == 'mean':
                if self.powder is None:
                    self.powder = np.sum(images, axis=0)
                else:
                    self.powder = np.sum(np.concatenate((self.powder[np.newaxis,:,:], images)), axis=0)
                
            else:
                raise ValueError("Invalid powder type, must be max or mean") 
            
            n_proc += images.shape[0] # at end of run, might not equal batch size            
            if images.shape[0] < batch_size: # reached end of the run
                break
                    
        if ptype == 'mean':
            self.powder /= float(n_proc)

        if plot:
            self.visualize_powder(self.powder)
        
        return self.powder

    def visualize_powder(self, image, vmin=-1e5, vmax=1e5,
                         output=None, figsize=12, dpi=300):
        """
        Visualize the powder image: the distribution of intensities as a histogram
        and the positive and negative-valued pixels on the assembled detector image.
        """
        fig = plt.figure(figsize=(figsize,figsize),dpi=dpi)
        gs = fig.add_gridspec(2,2)

        irow=0
        ax1 = fig.add_subplot(gs[irow,:2])
        ax1.grid()
        ax1.hist(image.flatten(), bins=100, log=True, color='black')
        ax1.set_title(f'histogram of pixel intensities in powder sum ',
                     fontdict={'fontsize': 8})

        irow+=1
        ax2 = fig.add_subplot(gs[irow,0])
        im = ax2.imshow(np.where(image>0,0,image),
                        cmap=plt.cm.gist_gray,
                        norm=colors.SymLogNorm(linthresh=1., linscale=1.,
                                               vmin=vmin, vmax=0.))
        ax2.axis('off')
        ax2.set_title(f'negative intensity pixels',
                     fontdict={'fontsize': 6})
        plt.colorbar(im)

        ax3 = fig.add_subplot(gs[irow,1])
        im = ax3.imshow(np.where(image<0,0,image),
                        cmap=plt.cm.gist_yarg,
                        norm=colors.SymLogNorm(linthresh=1., linscale=1.,
                                               vmin=0, vmax=vmax))
        ax3.axis('off')
        ax3.set_title(f'positive intensity pixels',
                     fontdict={'fontsize': 6})
        plt.colorbar(im)

        if output is not None:
            plt.savefig(output)

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
