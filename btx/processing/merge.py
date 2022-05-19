import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

class StreamtoMtz:
    
    """
    Wrapper for calling various CrystFEL functions for generating a mtz
    file from a stream file and reporting statistics. The output files,
    including hkl, mtz, dat, and figures, inherit nomenclature from the
    input stream file.
    """
    
    def __init__(self, input_stream, symmetry, taskdir, cell):
        
        self.stream = input_stream # file of unmerged reflections, str
        self.symmetry = symmetry # point group symmetry, str
        self.taskdir = taskdir # path for storing output, str
        self.cell = cell # pdb or CrystFEL cell file, str
        self._set_up()
        
    def _set_up(self):
        """
        Retrieve number of processors to run partialator and the temporary
        file to which to write the command to sbatch.
        """
        # handle parallel logic
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.nproc = comm.Get_size()
        if "NCORES" in os.environ:
            self.nproc = os.environ['NCORES']
        
        # make path to executable
        if "TMP_EXE" in os.environ:
            self.tmp_exe = os.environ['TMP_EXE']
        else:
            self.tmp_exe = os.path.join(self.taskdir ,f'merge.sh')
                
        # generate directories if they don't already exit
        self.hkl_dir = os.path.join(self.taskdir, "hkl")
        self.mtz_dir = os.path.join(self.taskdir, "mtz")
        self.fig_dir = os.path.join(self.taskdir, "figs")
        if self.rank == 0:
            for dirname in [self.hkl_dir, self.mtz_dir, self.fig_dir]:
                os.makedirs(dirname, exist_ok=True)

        # retrieve paths
        self.script_path = os.path.abspath(__file__)
        self.python_path = os.environ['WHICHPYTHON']
        self.prefix = os.path.basename(self.stream).split("stream")[0][:-1]
        self.outhkl = os.path.join(self.hkl_dir, f"{self.prefix}.hkl")

    def cmd_partialator(self, iterations=1, model='unity', min_res=None, push_res=None):
        """
        Write command to merge reflection data using CrystFEL's partialator.
        https://www.desy.de/~twhite/crystfel/manual-partialator.html.
        By default scaling and post-refinement are performed. This generates
        a .mtz file from a .stream file.
        
        Parameters
        ----------
        iterations : int
            number of cycles of scaling and post refinement
        model : str
            partiality model, unity or xsphere
        min_res : float
            resolution threshold for merging crystals in Angstrom
        push_res : float
            resolution threshold for merging reflections, up to push_res better than crystal's min_res
        """        
        if self.rank == 0:
            command=f"partialator -j {self.nproc} -i {self.stream} -o {self.outhkl} --iterations={iterations} -y {self.symmetry} --model={model}"

            # optionally add resolution thresholds for merging; otherwise, merge all
            if min_res is not None:
                command += f" --min-res={min_res}"
                if push_res is not None:
                    command += f" --push-res={push_res}"

            with open(self.tmp_exe, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"{command}\n")
            print(f"Merging executable written to {self.tmp_exe}")
            
    def cmd_compare_hkl(self, foms=['CC','Rsplit'], nshells=None, highres=None):
        """
        Run compare_hkl on half-sets to compute the dataset's figures of merit.
        Append a line to the executable to plot the results.
        
        Parameters
        ----------
        foms : list of str
            figures of merit to compute
        nshells : int
            number of resolution shells
        highres : float
            high-resolution cut-off
        """
        if self.rank == 0:
            for fom in foms:
                shell_file = os.path.join(self.hkl_dir, f"{self.prefix}_{fom}.dat")
                command = f'compare_hkl -y {self.symmetry} --fom {fom} {self.outhkl}1 {self.outhkl}2 -p {self.cell} --shell-file={shell_file}'
                if nshells is not None:
                    command += f" --nshells={nshells}"
                if highres is not None:
                    command += f" --highres={highres}"
                    
                with open(self.tmp_exe, 'a') as f:
                    f.write(f"{command}\n")
                print(f"Calculation for FOM {fom} appended to {self.tmp_exe}")
                
            # write command to run script from command line in report mode
            foms_args = " ".join(map(str,foms))
            command_report=f"{self.python_path} {self.script_path} -i {self.stream} --symmetry {self.symmetry} --cell {self.cell} --taskdir {self.taskdir} --fom {foms_args} --report"
            with open(self.tmp_exe, 'a') as f:
                f.write(f"{command_report}\n")
                
    def cmd_get_hkl(self):
        """
        Convert hkl to mtz format using CrystFEL's get_hkl tool:
        https://www.desy.de/~twhite/crystfel/manual-get_hkl.html
        """
        outmtz = os.path.join(self.mtz_dir, f"{self.prefix}.mtz")
        if self.rank == 0:
            command = f"get_hkl -i {self.outhkl} -o {outmtz} -p {self.cell} --output-format=mtz"
            with open(self.tmp_exe, 'a') as f:
                f.write(f"{command}\n")
        
    def report(self, foms=['CC','Rsplit'], update_url=None):
        """
        Summarize results: plot figures of merit and optionally report to elog.
        
        Parameters
        ----------
        foms : list of str
            figures of merit to compute
        update_url : str
            elog URL for posting progress update
        """
        if self.rank == 0:
            
            # plot each figure of merit
            for fom in foms:
                shell_file = os.path.join(self.hkl_dir, f"{self.prefix}_{fom}.dat")
                plot_file = os.path.join(self.fig_dir, f"{self.prefix}_{fom}.png")
                plot_shells_dat(shell_file, plot_file)
                        
            update_url = os.environ.get('JID_UPDATE_COUNTERS')
            if update_url is not None:
                print("Will report to elog...")
                        

def plot_shells_dat(shells_file, outfile=None):
    """
    Plot a shells file produced by CrystFEL's compare_hkl.
    
    Parameters
    ----------
    shells_file : str
        path to a CrystFEL shells.dat file
    outfile : str
        path for saving plot, optional
    """  
    shells = np.loadtxt(shells_file, skiprows=1)
    with open(shells_file) as f:
        lines = f.read() 
        header = lines.split('\n')[0]
    fom = header.split()[2]
    
    f, ax1 = plt.subplots(figsize=(6,4))

    ax1.plot(shells[:,0], shells[:,1], c='black')
    ax1.scatter(shells[:,0], shells[:,1], c='black')

    ticks = ax1.get_xticks()
    ax1.set_xticklabels(["{0:0.2f}".format(i) for i in 10/ticks])
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_xlabel("Resolution ($\mathrm{\AA}$)", fontsize=14)
    ax1.set_ylabel(f"{fom}", fontsize=14)
    
    if outfile is not None:
        f.savefig(outfile, dpi=300, bbox_inches='tight')

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    # global arguments
    parser.add_argument('-i', '--input_stream', required=True, type=str, help='Input stream file')
    parser.add_argument('--symmetry', required=True, type=str, help='Point group symmetry')
    parser.add_argument('--taskdir', required=True, type=str, help='Base directory for indexing results')
    parser.add_argument('--cell', required=True, type=str, help='File containing unit cell information (.pdb or .cell)')
    # arguments specific to partialator
    parser.add_argument('--model', default='unity', choices=['unity','xsphere'], type=str, help='Partiality model')
    parser.add_argument('--iterations', default=1, type=int, help='Number of cycles of scaling and post-refinement to perform')
    parser.add_argument('--min_res', required=False, type=float, help='Minimum resolution for crystal to be merged')
    parser.add_argument('--push_res', required=False, type=float, help='Maximum resolution beyond min_res for reflection to be merged')
    # arguments related to computing figures of merit and reporting
    parser.add_argument('--foms', required=False, type=str, nargs='+', help='Figures of merit to calculate')
    parser.add_argument('--nshells', required=False, type=float, default=10, help='Number of resolution shells for computing figures of merit')
    parser.add_argument('--highres', required=False, type=float, help='High resolution limit for computing figures of merit') 
    parser.add_argument('--report', help='Report indexing results to summary file and elog', action='store_true')
    parser.add_argument('--update_url', help='URL for communicating with elog', required=False, type=str)
    
    return parser.parse_args()

if __name__ == '__main__':
    
    params = parse_input()
    
    stream2mtz_obj = StreamtoMtz(params.input_stream, params.symmetry, params.taskdir, params.cell)
    
    if not params.report:
        stream2mtz_obj.cmd_partialator(iterations=params.iterations, model=params.model, min_res=params.min_res, push_res=params.push_res)
        stream2mtz_obj.cmd_compare_hkl(foms=params.foms, nshells=params.nshells, highres=params.highres)
        stream2mtz_obj.cmd_get_hkl()
    else:
        stream2mtz_obj.report(foms=params.foms, update_url=params.update_url)
