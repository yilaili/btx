import numpy as np
import argparse
import os

class Indexer:
    
    """ Index cxi files using indexamajig: https://www.desy.de/~twhite/crystfel/manual-indexamajig.html """

    def __init__(self, exp, run, det_type, tag, taskdir, geom, cell=None, int_rad='4,5,6', methods='mosflm',
                 tolerance='5,5,5,1.5', tag_cxi='', no_revalidate=True, multi=True, profile=True):
        
        # general paramters
        self.exp = exp
        self.run = run
        self.det_type = det_type
        
        # indexing parameters
        self.geom = geom # geometry file in CrystFEL format
        self.cell = cell # file containing unit cell information
        self.rad = int_rad # list of str, radii of integration
        self.methods = methods # str, indexing packages to run
        self.tolerance = tolerance # list of str, tolerances for unit cell comparison
        self.no_revalidate = no_revalidate # bool, skip validation step to omit iffy peaks
        self.multi = multi # bool, enable multi-lattice indexing
        self.profile = profile # bool, display timing data
        self._retrieve_paths(taskdir, tag_cxi, tag)
        self._parallel_logic()

    def _parallel_logic(self):
        """
        Retrieve number of processors to run indexamajig on. If running in 
        parallel, import mpi4py to ensure only first rank writes outfile.
        """
        self.nproc = os.environ['NCORES']
        if int(self.nproc) > 1:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            self.rank = comm.Get_rank()
        else:
            self.rank = 0

    def _retrieve_paths(self, taskdir, tag_cxi, tag):
        """
        Retrieve the paths for the input .lst and output .stream file 
        consistent with the btx analysis directory structure.
        
        Parameters
        ----------
        taskdir : str
            path to output folder for indexing results
        tag : str
            filename extension suffix
        """
        if ( tag_cxi != ''):
            tag_cxi = '_'+tag_cxi
        self.lst = os.path.join(taskdir ,f'r{self.run:04}/r{self.run:04}{tag_cxi}.lst')
        self.stream = os.path.join(taskdir, f'r{self.run:04}_{tag}.stream')
        if "TMP_EXE" in os.environ:
            self.tmp_exe = os.environ['TMP_EXE']
        else:
            self.tmp_exe = os.path.join(taskdir ,f'r{self.run:04}/index_r{self.run:04}.sh')
        
    def write_exe(self):
        """
        Write an indexing executable for submission to slurm.
        """     
        if self.rank == 0:
            command=f"indexamajig -i {self.lst} -o {self.stream} -j {self.nproc} -g {self.geom} --peaks=cxi --int-rad={self.rad} --indexing={self.methods} --tolerance={self.tolerance}"
            if self.cell is not None: command += f' --pdb={self.cell}'
            if self.no_revalidate: command += ' --no-revalidate'
            if self.multi: command += ' --multi'
            if self.profile: command += ' --profile'

            with open(self.tmp_exe, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"{command}\n")
            print(f"Indexing executable written to {self.tmp_exe}")

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', help='Experiment name', required=True, type=str)
    parser.add_argument('-r', '--run', help='Run number', required=True, type=int)
    parser.add_argument('-d', '--det_type', help='Detector name, e.g epix10k2M or jungfrau4M', required=True, type=str)
    parser.add_argument('--tag', help='Suffix extension for stream file', required=True, type=str)
    parser.add_argument('--tag_cxi', help='Tag to identify input CXI files', required=False, type=str)
    parser.add_argument('--taskdir', help='Base directory for indexing results', required=True, type=str)
    parser.add_argument('--geom', help='CrystFEL-style geom file', required=True, type=str)
    parser.add_argument('--cell', help='File containing unit cell information (.pdb or .cell)', required=False, type=str)
    parser.add_argument('--int_rad', help='Integration radii for peak, buffer and background regions', required=False, type=str, default='4,5,6')
    parser.add_argument('--methods', help='Indexing methods', required=False, type=str, default='xgandalf,mosflm,xds')
    parser.add_argument('--tolerance', help='Tolerances for unit cell comparison: a,b,c,ang', required=False, type=str, default='5,5,5,1.5')
    parser.add_argument('--no_revalidate', help='Skip validation step that omits peaks that are saturated, too close to detector edge, etc.', action='store_false')
    parser.add_argument('--multi', help='Enable multi-lattice indexing', action='store_false')
    parser.add_argument('--profile', help='Display timing data', action='store_false')
    
    return parser.parse_args()

if __name__ == '__main__':
    
    params = parse_input()
    indexer_obj = Indexer(exp=params.exp, run=params.run, det_type=params.det_type, taskdir=params.taskdir, geom=params.geom, 
                          cell=params.cell, int_rad=params.int_rad, methods=params.methods, tolerance=params.tolerance, tag_cxi=params.tag_cxi,
                          no_revalidate=params.no_revalidate, multi=params.multi, profile=params.profile)
    indexer_obj.write_exe()
