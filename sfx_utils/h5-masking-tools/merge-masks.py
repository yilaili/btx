#!/usr/bin/env python

import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt

#h5FileName = input("\ninput the first mask file to be merged: \n")
h5FileName = "ind.h5"
f = h5py.File(h5FileName, "r")
#data = f['entry']
data = f['data']
data1 = data['data']
tmask1 = np.array(data1[:,:])
mask1 = tmask1.copy()
f.close()

#h5FileName = input("\ninput the second mask file to be merged: \n")
h5FileName = "start.h5"
f = h5py.File(h5FileName, "r")
#data = f['entry']
data = f['data']
data2 = data['data']
tmask2 = np.array(data2[:,:])
mask2 = tmask2.copy()
f.close()

mask = mask1 * mask2

plt.imshow(mask, interpolation='nearest',norm=None)
plt.show()

#h5FileName = input("\nWhat would you like the merged file to be called?  \n")
h5FileName = "start.h5"
print("writing file...\n")
f = h5py.File(h5FileName,"w")
data = f.create_group("data");
data.create_dataset("data",data=mask);
f.close()


#h5FileName = input("\nIf you'd like the mask superimposed on an image, input a single frame file name:  \n")
#f = h5py.File(h5FileName, "r")
#data = f['entry']
#data = f['data']
#data = data['data']
#tim = np.array(data[:,:])
#im = tim.copy()
#f.close()

#plt.imshow(10000*im*mask,interpolation="nearest",norm=None)
#plt.show()
 
