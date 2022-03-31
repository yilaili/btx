#!/usr/bin/env python

# Sabine's script for masking pixels in an image based on an upper and a lower threshold.
# Last edited: 20220217

import h5py
import sys
import numpy as np

h5FileName = str(sys.argv[1])
f = h5py.File(h5FileName, "r")
data = f['data']
data = data['data']
tim = np.array(data[:, :])
im = tim.copy()
f.close()

upper_threshold = 50000
lower_threshold = -20
mask = np.ones((im.shape[0], im.shape[1]))

cold = np.ones((im.shape[0], im.shape[1]))
hot = np.ones((im.shape[0], im.shape[1]))
cold[np.where(im < lower_threshold)] = 0.0
hot[np.where(im > upper_threshold)] = 0.0
print("\ndead pixels:  " + str((im.shape[0]*im.shape[1])-np.sum(hot)-np.sum(cold)) + "\n")
mask = mask * hot * cold

maskFileName = "pixel-mask.h5"
print("\nwriting file pixel-mask.h5 ...\n")
f = h5py.File(maskFileName,"w")
data = f.create_group("data")
data.create_dataset("data", data=mask)
f.close()


