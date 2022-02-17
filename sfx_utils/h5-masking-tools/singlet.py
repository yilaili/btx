#!/usr/bin/env python

import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
import random

#h5FileName = "/bioxfel/data/2018/NSLS-2018-April/data/20180417/302282/pkjet03_e01/489/FMX001_1/pkjet03_e01_2640_data_000001.h5"
#h5FileName = input("\nInput a multi-event HDF5 file for extracting a single image: \n")
h5FileName= str(sys.argv[1])
f = h5py.File(h5FileName, "r")
data = f['entry_1']
data = data['data_1']
data = data['data']
num_im = random.randint(0,data.shape[0])
print("\n----- The random image that was picked is image #" + str(num_im) +" -----\n")
 
tim = np.array(data[num_im,:,:])
im = tim.copy()
f.close()

#H5FileName = input("\nAnd what should the single frame be named: \n")
#H5FileName = "single" + str(sys.argv[2])+ ".h5"
H5FileName = "single.h5"
print("\nwriting file...\n") 
f = h5py.File(H5FileName,"w")
data = f.create_group("data");
data.create_dataset("data",data=im);
f.close()

