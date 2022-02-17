#!/usr/bin/env python

import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt

#h5FileName = "/bioxfel/data/2018/NSLS-2018-April/analysis/scripts_nsls/single-mof01-000003_0.h5"
#h5FileName = input("\nPlease give me a single frame I can use to make the mask: \n")
h5FileName = "single.h5"
f = h5py.File(h5FileName, "r")
data = f['data']
#data = entry['data']
data = data['data']
tim = np.array(data[:,:])
im = tim.copy()
f.close()

#threshold = int(input("\nWhat would you like the ADU threshold for a hot pixel to be: \n"))
threshold = 50000
mask = np.ones((im.shape[0], im.shape[1]))

dead = np.ones((im.shape[0], im.shape[1]))
hot = np.ones((im.shape[0], im.shape[1]))
dead[np.where(im < -20)] = 0.0
hot[np.where(im > threshold )] = 0.0
print("\ndead pixels:  " + str((im.shape[0]*im.shape[1])-np.sum(dead)) + "\n")
mask = mask * dead * hot

#print("\nplotting the mask...\n")
#plt.imshow(mask,interpolation="nearest",norm=None)                     
#plt.show() 

#print("\nplotting the mask superimposed on an image...\n")
#plt.imshow(10000*im*mask,interpolation="nearest",norm=None)
#plt.show()


#maskFileName = input("\nThe mask file should be called: \n")
maskFileName = "ind.h5"
print("\nwriting file...\n") 
f = h5py.File(maskFileName,"w")
data = f.create_group("data")
data.create_dataset("data",data=mask)
f.close()


