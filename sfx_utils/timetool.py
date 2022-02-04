from psana_interface import *

class TimeTool:

    def __init__(self, exp, run, det_type):
        self.psi = PsanaInterface(exp=exp,
                                  run=run,
                                  det_type=det_type)
