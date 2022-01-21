import numpy as np
import matplotlib.pyplot as plt
from psana_interface import *

class RunDiagnostics:
    
    def __init__(self, exp, run, det_type):
        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type, track_timestamps=False)
        self.run_stats = dict()
        self.powder = None
        
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
        
    def compute_run_stats(self, n_images=1e6, batch_size=100, max_devs=50):
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
        """
        n_proc = 0
        while n_proc < n_images:
            try:
                images = self.psi.get_images(batch_size, assemble=False)
                for threshold in [None, max_devs]:
                    batch_stats = self.compute_batch_stats(images, max_devs=threshold)
                    self.wrangle_run_stats(batch_stats)
                n_proc += batch_size
                
            except:
                print("Reached the end of the run or requested number of images.")
                break
        return
    
    def plot_run_stats(self, tag='', output=None):
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
            
        return
