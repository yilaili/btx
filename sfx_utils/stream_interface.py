import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import xtal_utils

class StreamInterface:
    
    def __init__(self, input_file, cell_only=False):
        self.stream_file = input_file
        self.cell_only = cell_only
        self.stream_data = self.read_stream(input_file)
    
    def read_stream(self, input_file):
        """
        Read stream file. Function possibly adapted from CrystFEL.
        
        Parameters
        ----------
        input_file : string
            stream file to parse
        
        Returns
        -------
        stream_data : numpy.ndarray, shape (n_refl,14) or (n_refl,7)
            [crystal_num,a,b,c,alpha,beta,gamma,h,k,l,I,sigma(I),peak,background,res]
            or [crystal_num,a,b,c,alpha,beta,gamma] if self.cell_only=False
        """
        stream_data = []
        n_cryst = -1
        in_refl = False

        f = open(input_file)
        for line in f:
            if line.find("Cell parameters") != -1:
                cell = line.split()[2:5] + line.split()[6:9]
                cell = np.array(cell).astype(float)
                n_cryst+=1

                if self.cell_only:
                    stream_data.append(np.concatenate((np.array([n_cryst]), cell)))

            if not self.cell_only:
                if line.find("Reflections measured after indexing") != -1:
                    in_refl = True
                    continue

                if line.find("End of reflections") != -1:
                    in_refl = False
                    continue

                if in_refl:
                    if line.find("h    k    l") == -1:
                        reflection = np.array(line.split()[:7]).astype(float)
                        stream_data.append(np.concatenate((np.array([n_cryst]), cell, reflection, np.array([-1]))))
                        continue

        f.close()

        stream_data = np.array(stream_data)
        if not self.cell_only:
            stream_data[:,-1] = xtal_utils.compute_resolution(stream_data[:,1:7], stream_data[:,7:10])
                
        return stream_data
    
    def plot_peakogram(self, output=None):
        """
        Generate a peakogram of the stream data. Code from Frédéric Poitevin.
        
        Parameters
        ----------
        output : string, default=None
            if supplied, path for saving png of peakogram
        """
        if self.cell_only:
            print("Cannot plot peakogram because only cell parameters were extracted")
            return
              
        peak_res = self.stream_data[:,-1]
        peak_sum = self.stream_data[:,-3]
        peak_max = self.stream_data[:,-5]
        peak_sig = self.stream_data[:,-4]

        figsize = 8
        peakogram_bins = [500, 500]

        fig = plt.figure(figsize=(figsize, figsize), dpi=300)
        gs = fig.add_gridspec(2, 2)

        irow = 0
        ax1 = fig.add_subplot(gs[irow, 0:])
        ax1.set_title(f'Peakogram ({peak_res.shape[0]} reflections)')

        x, y = peak_res[peak_sum>0], np.log10(peak_sum[peak_sum>0])
        H, xedges, yedges = np.histogram2d(y, x, bins=peakogram_bins)
        im = ax1.pcolormesh(yedges, xedges, H, cmap='gray', norm=LogNorm())
        plt.colorbar(im)
        ax1.set_xlabel("1/d (${\mathrm{\AA}}$$^{-1}$)")
        ax1.set_ylabel("log(peak intensity)")

        irow += 1
        ax2 = fig.add_subplot(gs[irow, 0])
        im = ax2.hexbin(peak_sum, peak_max, gridsize=100, mincnt=1, norm=LogNorm(), cmap='gray')

        ax2.set_xlabel('sum in peak')
        ax2.set_ylabel('max in peak')
        plt.colorbar(im, label='No. reflections')

        ax3 = fig.add_subplot(gs[irow, 1])
        im = ax3.hexbin(peak_sum, peak_sig, gridsize=100, mincnt=1, norm=LogNorm(), cmap='gray')
        ax3.set_xlabel('sig(I)')
        ax3.set_yticks([])
        plt.colorbar(im, label='No. reflections')
        
        if output is not None:
            fig.savefig(output, dpi=300)
        
        return
    
    def plot_cell_parameters(self, output=None):
        """
        Plot histograms of the unit cell parameters, indicating the median
        cell axis and angle values.
        
        Parameters
        ----------
        output : string, default=None
            if supplied, path for saving png of unit cell parameters
        """
        unq_vals, unq_inds = np.unique(self.stream_data[:,0], return_index=True)
        
        f, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(12,5))

        labels = ["$a$", "$b$", "$c$", r"$\alpha$", r"$\beta$", r"$\gamma$"]
        for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6]):
            scale=10
            if i>2: scale=1
            vals = scale*self.stream_data[unq_inds,i+1]
            ax.hist(vals, bins=100, color='black')

            if i<3:
                ax.set_title(labels[i] + f"={np.median(vals):.3f}" + " ${\mathrm{\AA}}$")
            else:
                ax.set_title(labels[i] + f"={np.median(vals):.3f}" + "$^{\circ}$")
            if i == 0 or i == 3:
                ax.set_ylabel("No. crystals")

        f.subplots_adjust(hspace=0.4)
        
        if output is not None:
            f.savefig(output, dpi=300)
        
        return
