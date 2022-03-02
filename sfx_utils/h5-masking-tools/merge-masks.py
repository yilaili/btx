#!/usr/bin/env python

# Sabine's script for merging 2 masks in hdf5 format.
# Last edited: 20220217

import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt

h5FileName = str(sys.argv[1])
f = h5py.File(h5FileName, "r")
data = f['data']
data1 = data['data']
tmask1 = np.array(data1[:,:])
mask1 = tmask1.copy()
f.close()

h5FileName = str(sys.argv[2])
f = h5py.File(h5FileName, "r")
data = f['data']
data2 = data['data']
tmask2 = np.array(data2[:,:])
mask2 = tmask2.copy()
f.close()

mask = mask1 * mask2

plt.imshow(mask, interpolation='nearest', norm=None)
plt.show()

h5FileName = "merged-mask.h5"
print("writing file merged-mask.h5 ...\n")
f = h5py.File(h5FileName,"w")
data = f.create_group("data")
data.create_dataset("data", data=mask)
f.close()