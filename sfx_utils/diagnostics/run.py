import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from sfx_utils.interfaces.psana_interface import *

class RunDiagnostics:
    
    def __init__(self, exp, run, det_type):
        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type, track_timestamps=False)
        self.pixel_index_map = retrieve_pixel_index_map(self.psi.det.geometry(run))
        self.n_proc = 0 # number of images processed
        self.powders = dict() 
        self.run_stats = dict()
        
    def compute_powders(self, images):
        """
        Compute powder patterns, storing the average, max, and standard deivation. 

        Parameters
        ----------
        images : numpy.ndarray, 4d
            unassembled, calibrated images of shape (n_images, n_panels, n_x, n_y)
        """
        max_images = np.max(images, axis=0)
        sum_images = np.sum(images, axis=0)
        sqr_images = np.sum(np.square(images), axis=0)

        if not self.powders:
            for key,val in zip(['max','sum','sqr'],[max_images,sum_images,sqr_images]):
                self.powders[key] = val
        else:
            self.powders['sum'] += sum_images
            self.powders['sqr'] += sqr_images
            self.powders['max'] = np.max(np.concatenate((self.powders['max'][np.newaxis,:], max_images[np.newaxis,:])), axis=0)
            
        self.powders['avg'] = self.powders['sum'] / self.n_proc
        self.powders['std'] = np.sqrt(self.powders['sqr'] / self.n_proc - np.square(self.powders['avg']))

        return 
        
    def save_powders(self, output, assemble=True):
        """
        Store the powder patterns as individual numpy arrays to output.
        
        Parameters
        ----------
        output : str
            path to output directory
        assemble : bool
            if True, store images in assembled detector format
        """
        for key in ['max','avg','std']:
            if assemble:
                np.save(os.path.join(output, f"powder_{key}.npy"), 
                        assemble_image_stack_batch(self.powders[key], self.pixel_index_map))
            else:
                np.save(os.path.join(output, f"powder_{key}.npy"), self.powders[key])
        
        return
        
    def compute_batch_stats(self, images, max_devs=None):
        """
        Compute statistics for a batch of images. Even when images are pre-calibrated,
        some outliers may sneak through, so statistics are optionally computed after 
        removing outliers (pixels that exceed max_devs std deviations above the mean).
        
        Parameters
        ----------
        images : numpy.ndarray, 4d
            unassembled, calibrated images of shape (n_images, n_panels, n_x, n_y)
        max_devs : float, default=50
            number of standard deviations above mean to consider pixels outliers
            if None, do not perform additional outlier rejection.
        
        Returns
        -------
        stats : dict
            mean, standard deviation, median, max and min of each image
            with and without an additional outlier removal step
        """
        tag = ''
        if max_devs is not None:
            means, stds = np.mean(images, axis=(1,2,3)), np.std(images, axis=(1,2,3))
            images = np.where(np.abs(images - means[:,None,None,None]) < max_devs * stds[:,None,None,None], images, np.nan)
            tag = '_sel'
        
        stats = dict()
        stats['mean' + tag] = np.nanmean(images, axis=(1,2,3))
        stats['std_dev' + tag] = np.nanstd(images, axis=(1,2,3))
        stats['median' + tag] = np.nanmedian(images, axis=(1,2,3))
        stats['max' + tag] = np.nanmax(images, axis=(1,2,3))
        stats['min' + tag] = np.nanmin(images, axis=(1,2,3))
        
        return stats
    
    def wrangle_run_stats(self, batch_stats):
        """
        Store statistics from next batch of images.
        
        Parameters
        ----------
        batch_stats : dict
            dictionary of statistics for a batch of images
        """
        for key in batch_stats.keys():
            if key not in self.run_stats.keys():
                self.run_stats[key] = list(batch_stats[key])
            else:
                self.run_stats[key].extend(list(batch_stats[key]))
            
        return
        
    def compute_run_stats(self, n_images=1e6, batch_size=100, max_devs=50, powder_only=False):
        """
        Compute per-image statistics (mean, median, max, min, std deviation),
        with and without an additional step of outlier rejection.
        
        Parameters
        ----------
        n_images : int 
            number of images from run to process
        batch_size : int
            number of images per batch
        max_devs : float
            threshold for outlier removal (number of std deviations above mean)
        powder_only : bool
            if True, only compute the powder pattern
        """
        if batch_size > n_images:
            batch_size = n_images
            
        while self.n_proc < n_images:

            images = self.psi.get_images(batch_size, assemble=False)
            self.n_proc += images.shape[0]
            
            self.compute_powders(images)
            if not powder_only:
                for threshold in [None, max_devs]:
                    batch_stats = self.compute_batch_stats(images, max_devs=threshold)
                    self.wrangle_run_stats(batch_stats)

            if images.shape[0] < batch_size: # reached end of the run
                break

        return
        
    def visualize_powder(self, tag='max', vmin=-1e5, vmax=1e5,
                         output=None, figsize=12, dpi=300):
        """
        Visualize the powder image: the distribution of intensities as a histogram
        and the positive and negative-valued pixels on the assembled detector image.
        """

        image = assemble_image_stack_batch(self.powders[tag], self.pixel_index_map)
        
        fig = plt.figure(figsize=(figsize,figsize),dpi=dpi)
        gs = fig.add_gridspec(2,2)

        irow=0
        ax1 = fig.add_subplot(gs[irow,:2])
        ax1.grid()
        ax1.hist(image.flatten(), bins=100, log=True, color='black')
        ax1.set_title(f'histogram of pixel intensities in powder {tag}',
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
        
    def visualize_stats(self, tag='', output=None):
        """
        Plot trajectories of run statistics.
        
        Parameters
        ----------
        tag : string
            '' and '_sel' for pre and post outlier removal respectively
        output : string
            path for optionally saving plot to disk
        """
        f, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,8), sharex=True)

        keys = ['mean', 'max', 'min', 'std_dev']
        for ax,key in zip([ax1,ax2,ax3,ax4],keys):
            ax.plot(self.run_stats[key + tag], c='black')
            ax.set_ylabel(key, fontsize=12)
        
        ax.set_xlabel("Event", fontsize=12)
        if tag == '_sel':
            ax1.set_title("Run statistics after outlier removal")
        else:
            ax1.set_title("Run statistics")
            
        if output is not None:
            f.savefig(output, dpi=300)
    
