#!/usr/bin/env python

# Sabine's script for extracting a single image from a multi-event .cxi file (as created by cheetah).
# Last edited: 20220217

import h5py
import sys
import numpy as np
import random

h5FileName = str(sys.argv[1])
f = h5py.File(h5FileName, "r")
data = f['entry_1']
data = data['data_1']
data = data['data']
num_im = random.randint(0,data.shape[0])
print("\n----- The random image that was picked is image #" + str(num_im) +" -----\n")
 
tim = np.array(data[num_im, :, :])
im = tim.copy()
f.close()

H5FileName = "singlet.h5"
print("\nwriting file singlet.h5 ...\n")
f = h5py.File(H5FileName, "w")
data = f.create_group("data")
data.create_dataset("data", data=im)
f.close()