import numpy as np
import argparse
import itertools
import subprocess
import glob
import sys
import os
import time
import yaml
from btx.misc.shortcuts import AttrDict
from btx.misc.metrology import offset_geom
from btx.interfaces.stream_interface import *
from btx.processing.merge import wrangle_shells_dat

class Geoptimizer:
    """
    Class for refining the geometry. 
    """
    def __init__(self, queue, task_dir, scan_dir, runs, input_geom, dx_scan, dy_scan, dz_scan):
        
        self.queue = queue # queue to submit jobs to, str
        self.task_dir = task_dir # path to indexing directory, str
        self.scan_dir = scan_dir # path to scan directory, str
        self.runs = runs # run(s) to process, array
        
        self.timeout = 64800 # number of seconds to allow sbatch'ed jobs to run before exiting, float
        self.frequency = 5 # frequency in seconds to check on job completion, float
        self.crystfel_export="export PATH=/cds/sw/package/crystfel/crystfel-dev/bin:$PATH" 
        
        self._write_geoms(input_geom, dx_scan, dy_scan, dz_scan)
        self.logdir = os.path.join(self.scan_dir, "log")
        os.makedirs(self.logdir, exist_ok=True)
        
    def _write_geoms(self, input_geom, dx_scan, dy_scan, dz_scan):
        """
        Write a geometry file for each combination of offsets in dx,dy,dz.
        
        Parameters
        ----------
        input_geom : str
            path to original geometry
        dx_scan : numpy.array, 1d
            offsets along x (corner_x) to scan in pixels
        dy_scan : numpy.array, 1d
            offsets along y (corner_y) to scan in pixels
        dz_scan : numpy.array, 1d
            offsets along dz (coffset) to scan in mm
        """
        os.makedirs(os.path.join(self.scan_dir, "geom"), exist_ok=True)
        shifts_list = np.array(list(itertools.product(dx_scan,dy_scan,dz_scan)))

        for i,deltas in enumerate(shifts_list):
            output_geom = os.path.join(self.scan_dir, f"geom/shift{i}.geom")
            offset_geom(input_geom, output_geom, deltas[0], deltas[1], deltas[2])
            
        self.scan_results = np.zeros((len(shifts_list), 12))
        self.scan_results[:,:3] = shifts_list
        self.cols = ['dx', 'dy', 'dz']
            
    def _submit_bash_file(self, jobfile, jobname, command):
        """
        Submit a job submission script to run input command.
        
        Parameters
        ----------
        jobfile : str
            path to submission script
        jobname : str
            name of job to submit
        command : str
            bash command to run
        """
        with open(jobfile, "w") as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines(f"#SBATCH -p {self.queue}\n")
            fh.writelines(f"#SBATCH --job-name={jobname}\n")
            fh.writelines(f"#SBATCH --output={self.logdir}/{jobname}.out\n")
            fh.writelines(f"#SBATCH --output={self.logdir}/{jobname}.err\n")
            fh.writelines("#SBATCH --time=0:30:00\n")
            fh.writelines("#SBATCH --exclusive\n")
            fh.writelines(f"{self.crystfel_export}\n")
            fh.writelines(f"{command}\n")
        
        os.system(f"sbatch {jobfile}")

    def check_status(self, statusfile, jobnames, debug=False):
        """
        Check whether all launched jobs have completed.
        
        Parameters
        ----------
        statusfile : str
            path to file that lists completed jobnames
        jobnames : list of str
            list of all jobnames launched
        """
        done = False
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            if os.path.exists(statusfile) and not done:

                with open(statusfile, "r") as f:
                    lines = f.readlines()
                    finished = [l.strip('\n') for l in lines]
                    if set(finished) == set(jobnames):
                        print(f"All done: {jobnames}")
                        done = True
                        os.remove(statusfile)
                        time.sleep(self.frequency*5)
                        break                    
                time.sleep(self.frequency)

    def launch_indexing(self, params, ncores_per_job=4):
        """
        Launch indexing jobs.
        
        Parameters
        ----------
        params : btx.misc.shortcuts.AttrDict
            config.index object containing indexing parameters
        n_cores_per_job : int
            number of cores per indexing job
        """      
        jobnames = list()        
        statusfile = os.path.join(self.scan_dir,"status.sh")

        if params.get('tag_cxi') is not None :
            if ( params.tag_cxi != '' ) and ( params.tag_cxi[0]!='_' ):
                tag_cxi = '_'+params.tag_cxi
        else:
            tag_cxi = ''

        for run in self.runs:
            
            os.makedirs(os.path.join(self.scan_dir, f"r{run:04}"), exist_ok=True)
            lst_file = os.path.join(self.task_dir ,f'r{run:04}/r{run:04}{tag_cxi}.lst')
            
            for num in range(self.scan_results.shape[0]):
                
                jobname = f"r{run}_g{num}"
                jobfile = os.path.join(self.scan_dir,"temp.sh")
                stream = os.path.join(self.scan_dir, f'r{run:04}/r{run:04}_g{num}.stream')
                gfile = os.path.join(self.scan_dir, f"geom/shift{num}.geom")
                
                command=f"indexamajig -i {lst_file} -o {stream} -j {ncores_per_job} -g {gfile} --peaks=cxi --int-rad={params.int_radius} --indexing={params.methods} --tolerance={params.tolerance}"
                if params.cell is not None: command += f' --pdb={params.cell}'
                if params.no_revalidate: command += ' --no-revalidate'
                if params.multi: command += ' --multi'
                if params.profile: command += ' --profile'
                command += '\n'
                
                command += f"echo {jobname} | tee -a {statusfile}\n"

                self._submit_bash_file(jobfile, jobname, command)
                jobnames.append(jobname)

        self.check_status(statusfile, jobnames)

    def launch_stream_analysis(self, cell_file):
        """
        Compute the number of indexed crystals and standard deviations of unit
        cell parameters and concatenate, with one stream file per geometry file. 
        
        Parameters
        ----------
        cell_file : str
            path to cell file used during indexing
        """
        celldir = os.path.join(self.scan_dir, "cell")
        os.makedirs(celldir, exist_ok=True)

        for num in range(self.scan_results.shape[0]):
            stream_files = os.path.join(self.scan_dir, f"r*/r*_g{num}.stream")
            st = StreamInterface(glob.glob(stream_files), cell_only=True)
            write_cell_file(st.cell_params, os.path.join(celldir, f"g{num}.cell"), input_file=cell_file)
            stream_cat = os.path.join(self.scan_dir, f'g{num}.stream')
            os.system(f"cat {stream_files} >> {stream_cat}")
            
            self.scan_results[num,3:10] = np.append(np.array(len(st.unq_inds)), st.cell_params_std) # num indexed, cell std devs
        
        self.cols.extend(['n_indexed', 'a', 'b', 'c', 'alpha', 'beta', 'gamma'])
            
    def launch_merging(self, params, ncores_per_job=4):
        """
        Launch merging and hkl stats jobs.
        
        Parameters
        ----------
        params : btx.misc.shortcuts.AttrDict
            config.merge dictinoary containing merging parameters
        n_cores_per_job : int
            number of cores per indexing job
        queue : str
            queue for submitting indexing jobs
        """
        mergedir = os.path.join(self.scan_dir, "merge")
        os.makedirs(mergedir, exist_ok=True)
        for file_extension in ["*.hkl*", "*.dat"]:
            filelist = glob.glob(os.path.join(mergedir, file_extension))
            if len(filelist) > 0:
                for filename in filelist:
                    os.remove(filename)

        jobnames = list()
        statusfile = os.path.join(self.scan_dir,"status.sh")
        
        for num in range(self.scan_results.shape[0]):
                
            jobname = f"g{num}"
            jobfile = os.path.join(self.scan_dir,"temp.sh")
            instream = os.path.join(self.scan_dir, f'g{num}.stream')
            outhkl = os.path.join(mergedir, f"g{num}.hkl")
            cellfile = os.path.join(self.scan_dir, f"cell/g{num}.cell")
            
            command=f"partialator -j {ncores_per_job} -i {instream} -o {outhkl} --iterations={params.iterations} -y {params.symmetry} --model={params.model}"
            if params.get('min_res') is not None:
                command += f" --min-res={min_res}"
                if params.get('push_res') is not None:
                    command += f" --push-res={push_res}"
            command += '\n'
            
            for fom in params.foms.split(' '):
                shell_file = os.path.join(mergedir, f"g{num}_{fom}.dat")
                command += f'compare_hkl -y {params.symmetry} --fom {fom} {outhkl}1 {outhkl}2 -p {cellfile} --shell-file={shell_file} --nshells=1'
                if params.get('highres') is not None:
                    command += f" --highres={params.highres}"
                command += '\n'

            command += f"echo {jobname} | tee -a {statusfile}\n"
            #command += f"echo {jobname} >> {statusfile}\n"

            self._submit_bash_file(jobfile, jobname, command)
            jobnames.append(jobname)
            time.sleep(self.frequency)
            
        self.check_status(statusfile, jobnames, debug=True)
        self.extract_stats(mergedir, params.foms.split(' '))
        
    def extract_stats(self, mergedir, foms):
        """
        Extract the overall figure of merit statistics.
        
        Parameters
        ----------
        mergedir : str
            directory containing results of crystfel's compare_hkl
        foms : list of str
            figures of merit that were calculated
        """
        for col,fom in enumerate(foms):
            self.cols.append(fom)
            overall_stat = np.zeros(self.scan_results.shape[0])
            
            for num in range(self.scan_results.shape[0]):
                shells_file = os.path.join(mergedir, f"g{num}_{fom}.dat")
                fom_from_shell_file, stat = wrangle_shells_dat(shells_file)
                overall_stat[num] = stat
            self.scan_results[:,10+col] = overall_stat

    def save_results(self, savepath=None):
        """
        Save the results of the scan to a text file.

        Parameters
        ----------
        savepath : str
            text file to save results to
        """
        if savepath is None:
            savepath = os.path.join(self.scan_dir, "results.txt")

        fmt=['%.2f', '%.2f', '%.4f', '%d', '%.5f', '%.5f', '%.5f', '%.5f', '%.5f', '%.5f', '%.2f', '%.2f']
        np.savetxt(savepath, self.scan_results, header=' '.join(self.cols), fmt=fmt)

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help='Config file in yaml format')
    parser.add_argument('-g', '--geom', required=True, type=str, help='CrystFEL .geom file with starting metrology')
    parser.add_argument('-q', '--queue', required=True, type=str, help='Queue to submit jobs to')
    parser.add_argument('--runs', required=True, type=int, nargs=2, help='Runs to process: [start,end)')
    parser.add_argument('--dx', required=False, type=float, nargs=3, default=[0,0,1], help='xoffset translational scan in pixels: start, stop, n_points')
    parser.add_argument('--dy', required=False, type=float, nargs=3, default=[0,0,1], help='yoffset translational scan in pixels: start, stop, n_points')
    parser.add_argument('--dz', required=False, type=float, nargs=3, default=[0,0,1], help='coffset (distance) scan in mm: start, stop, n_points')
    parser.add_argument('--merge_only', help='Runs already indexed; perform merging only', action='store_true')
    
    return parser.parse_args()

if __name__ == '__main__':

    params = parse_input()
    with open(params.config, "r") as config_file:
        config = AttrDict(yaml.safe_load(config_file))

    taskdir = os.path.join(config.setup.root_dir, "index")
    scandir = os.path.join(config.setup.root_dir, "scan")
    os.makedirs(scandir, exist_ok=True)

    geopt = Geoptimizer(params.queue,
                        taskdir,
                        scandir,
                        np.arange(params.runs[0], params.runs[1]),
                        params.geom,
                        np.linspace(params.dx[0], params.dx[1], int(params.dx[2])),
                        np.linspace(params.dy[0], params.dy[1], int(params.dy[2])),
                        np.linspace(params.dz[0], params.dz[1], int(params.dz[2])))
    if not params.merge_only:
        geopt.launch_indexing(config.index)
        geopt.launch_stream_analysis(config.index.cell)
    geopt.launch_merging(config.merge)
    geopt.save_results()
