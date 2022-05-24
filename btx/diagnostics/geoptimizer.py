import numpy as np
import itertools
import subprocess
import os
import time
import yaml
from btx.misc.shortcuts import AttrDict
from btx.misc.metrology import offset_geom
from btx.interfaces.stream_interface import *

class Geoptimizer:
    """
    Class for refining the geometry. 
    """
    def __init__(self, task_dir, scan_dir, input_geom, runs, dx_scan, dy_scan, dz_scan):
        self.runs = runs # run(s) to process
        self.task_dir = task_dir # path to indexing directory
        self.scan_dir = scan_dir # path to scan directory
        self._write_geoms(input_geom, dx_scan, dy_scan, dz_scan)
        self.metrics = dict()
        
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

        self.d_geom = dict()
        for i,deltas in enumerate(shifts_list):
            output_geom = os.path.join(self.scan_dir, f"geom/shift{i}.geom")
            offset_geom(input_geom, output_geom, deltas[0], deltas[1], deltas[2])
            self.d_geom[output_geom] = deltas
            
    def _check_for_completion(self, jobnames):
        """
        Check whether the jobs listed in jobnames have finished.
        
        Parameters
        ----------
        jobnames : list of str
            names of jobs whose status to query
            
        Returns
        -------
        jobnames : list of str
            names of jobs still running on queue
        """
        for job in jobnames:
            process = subprocess.Popen([f'sacct --name {job} -X -n -o state'],
                           stdout = subprocess.PIPE, 
                           stderr = subprocess.PIPE,
                           text = True,
                           shell = True)

            std_out, std_err = process.communicate()
            if 'COMPLETED' in std_out and 'RUNNING' not in std_out:
                jobnames.remove(job)
                
        return jobnames
            
    def launch_indexing(self, params, ncores_per_job=4, queue='psanaq'):
        """
        Launch indexing jobs.
        
        Parameters
        ----------
        params : btx.misc.shortcuts.AttrDict
            config.index object containing indexing parameters
        n_cores_per_job : int
            number of cores per indexing job
        queue : str
            queue for submitting indexing jobs
        """
        
        logdir = os.path.join(self.scan_dir, "log")
        os.makedirs(logdir, exist_ok=True)
        crystfel_export="export PATH=/cds/sw/package/crystfel/crystfel-dev/bin:$PATH"
        jobnames = list()
        
        for run in self.runs:
            
            os.makedirs(os.path.join(self.scan_dir, f"r{run:04}"), exist_ok=True)
            lst_file = os.path.join(self.task_dir ,f'r{run:04}/r{run:04}.lst')
            
            for num,gfile in enumerate(self.d_geom.keys()):
                
                jobname = f"r{run}_g{num}"
                job_file = os.path.join(self.scan_dir,"temp.sh")
                stream = os.path.join(self.scan_dir, f'r{run:04}/r{run:04}_g{num}.stream')
                
                command=f"indexamajig -i {lst_file} -o {stream} -j {ncores_per_job} -g {gfile} --peaks=cxi --int-rad={params.int_radius} --indexing={params.methods} --tolerance={params.tolerance}"
                if params.cell is not None: command += f' --pdb={params.cell}'
                if params.no_revalidate: command += ' --no-revalidate'
                if params.multi: command += ' --multi'
                if params.profile: command += ' --profile'

                with open(job_file, "w") as fh:
                    fh.writelines("#!/bin/bash\n")
                    fh.writelines(f"#SBATCH -p {queue}\n")
                    fh.writelines(f"#SBATCH --job-name={jobname}\n")
                    fh.writelines(f"#SBATCH --output={logdir}/{jobname}.out\n")
                    fh.writelines(f"#SBATCH --output={logdir}/{jobname}.err\n")
                    fh.writelines("#SBATCH --time=0:30:00\n")
                    fh.writelines("#SBATCH --exclusive\n")
                    fh.writelines(f"{crystfel_export}\n")
                    fh.writelines(f"{command}\n")
                    
                os.system("sbatch %s" %job_file)
                jobnames.append(jobname)
                
        t = time.time()
        while True:
            if time.time() - t > 5*3600:
                break
            jobnames = self._check_for_completion(jobnames)
            if len(jobnames) == 0:
                print("All indexing jobs completed!")
                break
            time.sleep(5)                

    def perform_stream_analysis(self, cell_file):
        """
        
        """
        celldir = os.path.join(self.scan_dir, "cell")
        os.makedirs(celldir, exist_ok=True)

        for num,gfile in enumerate(self.d_geom.keys()):
            stream_files = [os.path.join(self.scan_dir, f"r{run:04}/r{run:04}_g{num}.stream") for run in self.runs]
            st = StreamInterface(stream_files, cell_only=True)
            write_cell_file(st.cell_params, os.path.join(celldir, f"g{num}.cell"), input_file=cell_file)
            self.metrics[gfile] = np.append(np.array(len(st.unq_inds)), st.cell_params_std) # num indexed, cell std devs
            stream_cat = os.path.join(self.scan_dir, f'g{num}.stream')
            os.system(f"cat {stream_files} >> {stream_cat}")

if __name__ == '__main__':

    dx = np.arange(-0.5,0.5,0.5)
    dy = np.arange(-0.005,0.005,0.005)
    dz = np.arange(-0.005,0.005,0.005)

    task_dir = "/cds/data/psdm/mfx/mfxp22820/scratch/apeck/index"
    scan_dir = "/cds/data/psdm/mfx/mfxp22820/scratch/apeck/scan"
    input_geom = "/cds/data/psdm/mfx/mfxp22820/scratch/apeck/geom/r0057.geom"

    config_filepath = "/cds/home/a/apeck/btx/tutorial/ap_mfxp22820.yaml"
    with open(config_filepath, "r") as config_file:
        config = AttrDict(yaml.safe_load(config_file))

    geopt = Geoptimizer(task_dir, scan_dir, input_geom, np.arange(26,28), dx, dy, dz)
    geopt.launch_indexing(config.index, queue='ffbh3q')
    geopt.perform_stream_analysis(config.index.cell)

    for key,val in geopt.metrics.items():
        print(key,val,"\n")
