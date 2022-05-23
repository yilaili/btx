import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from btx.misc.xtal import compute_resolution
from mpi4py import MPI
import glob
import argparse
import os
import requests

class StreamInterface:
    
    def __init__(self, input_files, cell_only=False):
        self.cell_only = cell_only # bool, if True only extract unit cell params
        self.input_files = input_files # list of stream file(s)
        self.stream_data, self.file_limits = self.read_all_streams(self.input_files)
        self.compute_mean_cell()
    
    def read_all_streams(self, input_files):
        """
        Read stream file(s), with files distributed across multiple ranks
        if available.
        
        Parameters
        ----------
        input_files : list of str
            stream file(s) to parse
        
        Returns
        -------
        stream_data : numpy.ndarray, shape (n_refl,16) or (n_refl,8)
            [crystal_num,a,b,c,alpha,beta,gamma,h,k,l,I,sigma(I),peak,background,res]
            or [crystal_num,a,b,c,alpha,beta,gamma] if self.cell_only=False        
        file_limits : numpy.ndarray, shape (n_files)
            indices of stream_data's first dimension that indicate start/end of each file
        """
        # processing all files given to each rank
        stream_data_rank = []
        input_sel = self.distribute_streams(input_files) 
        if len(input_sel) != 0:
            for ifile in input_sel:
                single_stream_data = self.read_stream(ifile)
                if len(single_stream_data) == 0: # no crystals in stream file
                    if self.cell_only: 
                        single_stream_data = np.empty((0,8))
                    else: 
                        single_stream_data = np.empty((0,16))
                stream_data_rank.append(single_stream_data)
            file_limits_rank = np.array([sdr.shape[0] for sdr in stream_data_rank])
            stream_data_rank = np.vstack(stream_data_rank)  
        else: # no files assigned to this rank
            if self.cell_only:
                stream_data_rank = np.empty((0,8))
            else:
                stream_data_rank = np.empty((0,16))
            file_limits_rank = np.empty(0)
        
        # amassing files from different ranks
        stream_data = self.comm.gather(stream_data_rank, root=0)
        file_limits = self.comm.gather(file_limits_rank, root=0)
        if self.rank == 0:
            stream_data = np.vstack(stream_data)
            file_limits = np.concatenate(file_limits)
            file_limits = np.append(np.array([0]), np.cumsum(file_limits))
        return stream_data, file_limits
        
    def distribute_streams(self, input_files):
        """
        Evenly distribute stream files among available ranks.
        
        Parameters
        ----------
        input_files : list of str
            list of input stream files to read
    
        Returns
        -------
        input_sel : list of str
            select list of input stream files for this rank
        """
        # set up MPI object
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size() 
        
        # divvy up files
        n_files = len(input_files)
        split_indices = np.zeros(self.size)
        for r in range(self.size):
            num_per_rank = n_files // self.size
            if r < (n_files % self.size):
                num_per_rank += 1
            split_indices[r] = num_per_rank
        split_indices = np.append(np.array([0]), np.cumsum(split_indices)).astype(int) 
        return input_files[split_indices[self.rank]:split_indices[self.rank+1]]
    
    def read_stream(self, input_file):
        """
        Read a single stream file. Function possibly adapted from CrystFEL.
        
        Parameters
        ----------
        input_file : str
            stream file to parse
        
        Returns
        -------
        single_stream_data : numpy.ndarray, shape (n_refl,14) or (n_refl,7)
            [crystal_num,a,b,c,alpha,beta,gamma,h,k,l,I,sigma(I),peak,background,res]
            or [crystal_num,a,b,c,alpha,beta,gamma] if self.cell_only=False
        """
        single_stream_data = []
        n_cryst, n_chunk = -1, -1
        in_refl = False

        f = open(input_file)
        for lc,line in enumerate(f):
            if line.find("Begin chunk") != -1:
                n_chunk += 1
                if in_refl:
                    in_refl = False
                    print(f"Warning! Line {lc} associated with chunk {n_chunk} is problematic: {line}")
            
            if line.find("Cell parameters") != -1:
                cell = line.split()[2:5] + line.split()[6:9]
                cell = np.array(cell).astype(float)
                n_cryst+=1

                if self.cell_only:
                    single_stream_data.append(np.concatenate((np.array([n_chunk, n_cryst]), cell)))

            if not self.cell_only:
                if line.find("Reflections measured after indexing") != -1:
                    in_refl = True
                    continue

                if line.find("End of reflections") != -1:
                    in_refl = False
                    continue

                if in_refl:
                    if line.find("h    k    l") == -1:
                        try:
                            reflection = np.array(line.split()[:7]).astype(float)
                            single_stream_data.append(np.concatenate((np.array([n_chunk, n_cryst]), cell, reflection, np.array([-1]))))
                        except ValueError:
                            print(f"Couldn't parse line {lc}: {line}")
                        continue

        f.close()

        single_stream_data = np.array(single_stream_data)
        if not self.cell_only:
            if len(single_stream_data) == 0:
                print(f"Warning: no indexed reflections found in {input_file}!")
            else:
                single_stream_data[:,-1] = compute_resolution(single_stream_data[:,2:8], single_stream_data[:,8:11])
                
        return single_stream_data
    
    def compute_mean_cell(self):
        """ 
        Compute the mean unit cell parameters: [a,b,c,alpha,beta,gamma] in A/degrees. 
        Since the crystal number column (self.stream_data[:,1]) resets to 0 for each
        input stream file, a self.ncryst variable is also created to track the crystal 
        index, ignoring which file each data entry comes from.
        """

        ncryst = self.stream_data[:,1].copy()
        next_cryst_idx = np.where(np.diff(ncryst)<0)[0] + 1
        next_cryst_idx = np.append(next_cryst_idx, np.array(len(ncryst)))
        ncryst_to_add = ncryst[next_cryst_idx-1]

        for idx in range(len(next_cryst_idx)-1):
            ncryst[next_cryst_idx[idx]:] += ncryst_to_add[idx] + 1
        self.ncryst=ncryst

        unq_vals, self.unq_inds = np.unique(self.ncryst, return_index=True)
        unq_cells = self.stream_data[self.unq_inds,2:8]
        self.cell_params = np.mean(unq_cells, axis=0)
        self.cell_params[:3]*=10 # convert unit cell from nm to Angstrom    
        self.cell_params_std = np.std(unq_cells, axis=0)
        self.cell_params_std[:3]*=10
        
    def get_cell_parameters(self):
        """ Retrieve unit cell parameters: [a,b,c,alpha,beta,gamma] in A/degrees. """
        return self.stream_data[:,2:8] * np.array([10.,10.,10.,1,1,1])
    
    def get_peak_res(self):
        """ Retrieve resolution of peaks in 1/Angstrom from self.stream_data. """
        if self.cell_only:
            print("Reflections were not extracted")
        else:
            return self.stream_data[:,-1]
    
    def get_peak_sumI(self):
        """ Retrieve summed intensity of peaks from self.stream_data. """
        if self.cell_only:
            print("Reflections were not extracted")
        else:
            return self.stream_data[:,-5]

    def get_peak_maxI(self):
        """ Retrieve max intensity of peaks from self.stream_data. """
        if self.cell_only:
            print("Reflections were not extracted")
        else:
            return self.stream_data[:,-3]
    
    def get_peak_sigI(self):
        """ Retrieve peaks' standard deviations from self.stream_data. """
        if self.cell_only:
            print("Reflections were not extracted")
        else:
            return self.stream_data[:,-4]
    
    def plot_peakogram(self, output=None, plot_Iogram=False):
        """
        Generate a peakogram of the stream data.
        
        Parameters
        ----------
        output : string, default=None
            if supplied, path for saving png of peakogram
        plot_Iogram : boolean, default=False
            if True, plot integrated rather than max intensities in the peakogram
        """
        if self.cell_only:
            print("Cannot plot peakogram because only cell parameters were extracted")
            return
              
        peak_res = self.get_peak_res()
        peak_sig = self.get_peak_sigI()
        if not plot_Iogram:
            peak_sum = self.get_peak_sumI()
            peak_max = self.get_peak_maxI()
            xlabel, ylabel = "sum", "max"
        # if Iogram, swap peak_sum and peak_max
        else:
            peak_sum = self.get_peak_maxI()
            peak_max = self.get_peak_sumI()
            xlabel, ylabel = "max", "sum"

        figsize = 8
        peakogram_bins = [500, 500]

        fig = plt.figure(figsize=(figsize, figsize), dpi=300)
        gs = fig.add_gridspec(2, 2)

        irow = 0
        ax1 = fig.add_subplot(gs[irow, 0:])
        ax1.set_title(f'Peakogram ({peak_res.shape[0]} reflections)')

        x, y = peak_res[peak_max>0], np.log10(peak_max[peak_max>0]) # can be negative for Iogram
        H, xedges, yedges = np.histogram2d(y, x, bins=peakogram_bins)
        im = ax1.pcolormesh(yedges, xedges, H, cmap='gray', norm=LogNorm())
        plt.colorbar(im)
        ax1.set_xlabel("1/d (${\mathrm{\AA}}$$^{-1}$)")
        ax1.set_ylabel(f"log(peak intensity) - {ylabel}")

        irow += 1
        ax2 = fig.add_subplot(gs[irow, 0])
        im = ax2.hexbin(peak_sum, peak_max, gridsize=100, mincnt=1, norm=LogNorm(), cmap='gray')

        ax2.set_xlabel(f'{xlabel} in peak')
        ax2.set_ylabel(f'{ylabel} in peak')

        ax3 = fig.add_subplot(gs[irow, 1])
        im = ax3.hexbin(peak_sig, peak_max, gridsize=100, mincnt=1, norm=LogNorm(), cmap='gray')
        ax3.set_xlabel('sig(I)')
        ax3.set_yticks([])
        plt.colorbar(im, label='No. reflections')
        
        if output is not None:
            fig.savefig(output, dpi=300)
    
    def plot_cell_parameters(self, output=None):
        """
        Plot histograms of the unit cell parameters, indicating the median
        cell axis and angle values.
        
        Parameters
        ----------
        output : string, default=None
            if supplied, path for saving png of unit cell parameters
        """
        f, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(12,5))

        labels = ["$a$", "$b$", "$c$", r"$\alpha$", r"$\beta$", r"$\gamma$"]
        for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6]):
            scale=10
            if i>2: scale=1
            vals = scale*self.stream_data[self.unq_inds,i+2] # first two cols are chunk,crystal
            ax.hist(vals, bins=100, color='black')

            if i<3:
                ax.set_title(labels[i] + f"={np.mean(vals):.3f}" + " ${\mathrm{\AA}}$")
            else:
                ax.set_title(labels[i] + f"={np.mean(vals):.3f}" + "$^{\circ}$")
            if i == 0 or i == 3:
                ax.set_ylabel("No. crystals")

        f.subplots_adjust(hspace=0.4)
        
        if output is not None:
            f.savefig(output, dpi=300)

    def report(self, update_url=None):
        """
        Summarize the cell parameters and optionally report to the elog.
    
        Parameters
        ----------
        update_url : str
            elog URL for posting progress update
        """
        # write summary file
        summary_file = os.path.join(os.path.dirname(self.input_files[0]), "stream.summary")
        with open(summary_file, 'w') as f:
            f.write("Cell mean: " + " ".join(f"{self.cell_params[i]:.3f}" for i in range(self.cell_params.shape[0])) + "\n")
            f.write("Cell std: " + " ".join(f"{self.cell_params_std[i]:.3f}" for i in range(self.cell_params.shape[0])) + "\n")
                        
        # report to elog
        update_url = os.environ.get('JID_UPDATE_COUNTERS')
        if update_url is not None:
            labels = ["a", "b", "c", "alpha", "beta", "gamma"]
            elog_json = [{"key": labels[i], "value": f"{self.cell_params[i]:.3f} +/- {self.cell_params_std[i]:.3f}"} for i in range(len(labels))]
            requests.post(update_url, json=elog_json)
    
    def copy_from_stream(self, stream_file, indices, indices_chunks, output):
        """
        Add the indicated crystals from the input to the output stream.
        
        Parameters
        ----------
        stream_file : str
            input stream file
        indices : numpy.ndarray, 1d
            indices of reflections (or crystals if self.cell_only) to retain
        indices_chunks : numpy.ndarray, 1d
            indices of chunks (crystals) to retain
        output : string
            path to output .stream file     
        """

        f_out = open(output, "a")
        f_in = open(stream_file)
        
        n_chunk = -1
        hkl_counter = -1
        in_header = True
        current_chunk = False
        in_refl = False

        for line in f_in:

            # copy header of stream file before first chunk begins
            if in_header:
                if line.find("Indexing methods") != -1:
                    in_header = False
                f_out.write(line)

            # in chunk territory...
            if line.find("Begin chunk") != -1:
                n_chunk += 1
                if n_chunk in indices_chunks:
                    current_chunk = True            
            if current_chunk:
                if in_refl:
                    if (hkl_counter+1 in indices) or (self.cell_only):
                    #if hkl_counter+1 in indices:
                        if line.find("End of reflections") == -1:
                            f_out.write(line)
                else:
                    f_out.write(line)
            if line.find("End chunk") != -1:
                current_chunk = False

            # count reflections
            if line.find("h    k    l") != -1:
                in_refl = True
                continue
            if line.find("End of reflections") != -1:
                if current_chunk:
                    f_out.write(line)
                in_refl = False
            if in_refl:
                hkl_counter += 1

        f_in.close()
        f_out.close()
        
    def write_stream(self, all_indices, output):
        """
        Write a new stream file from a selection of crystals or reflections of
        the original stream file(s).
        
        Parameters
        ----------
        all_indices : numpy.ndarray, 1d
            indices of self.stream_data to retain
        output : string
            path to output .stream file     
        """
        for nf,infile in enumerate(self.input_files):
            lower, upper = self.file_limits[nf], self.file_limits[nf+1]
            idx = np.where((all_indices>=lower) & (all_indices<upper))[0]
            sel_indices = all_indices[idx] - lower
            if len(sel_indices) > 0:
                print(f"Copying {len(sel_indices)} chunks (or reflections) from file {infile}")
                sel_chunks = np.unique(self.stream_data[all_indices[idx]][:,0]).astype(int)
                self.copy_from_stream(infile, sel_indices, sel_chunks, output)

def write_cell_file(cell, output_file, input_file=None):
    """
    Write a new CrystFEL-style cell file with the same lattice type as
    the input file (or primitive triclinic if none given) and the cell 
    parameters changed to input cell.
    
    Parameters
    ----------
    cell : np.array, 1d
        unit cell parameters [a,b,c,alpha,beta,gamma], in Å/degrees
    output_file : str
        output CrystFEL unit cell file
    input_file : str
        input CrystFEL unit cell file, optional
    """
    from itertools import islice

    if input_file is not None:
        with open(input_file, "r") as in_cell:
            header = list(islice(in_cell, 4))
    else:
        header = ['CrystFEL unit cell file version 1.0\n',
                  '\n',
                  'lattice_type = triclinic\n',
                  'centering = P\n']

    outfile = open(output_file, "w")
    for item in header:
        outfile.write(item)
    outfile.write(f'a = {cell[0]:.3f} A\n')
    outfile.write(f'b = {cell[1]:.3f} A\n')
    outfile.write(f'c = {cell[2]:.3f} A\n')
    outfile.write(f'al = {cell[3]:.3f} deg\n')
    outfile.write(f'be = {cell[4]:.3f} deg\n')
    outfile.write(f'ga = {cell[5]:.3f} deg\n')
    outfile.close()

#### For command line use ####
            
def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', help='Input stream files in glob-readable format', required=True, type=str)
    parser.add_argument('-o', '--outdir', help='Output directory for peakogram and cell plots', required=True, type=str)
    parser.add_argument('--cell_only', help='Only read unit cell parameters, not reflections', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':

    params = parse_input()
    st = StreamInterface(input_files=glob.glob(params.inputs), cell_only=params.cell_only)
    if st.rank == 0:
        st.plot_cell_parameters(output=os.path.join(params.outdir, "cell_distribution.png"))
        if not params.cell_only:
            st.plot_peakogram(output=os.path.join(params.outdir, "peakogram.png"))
        st.report(output=os.path.join(params.outdir, "cell.summary"))
