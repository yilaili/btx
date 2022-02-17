#!/usr/bin/env python

#Sabine's script for drawing a (arguably deflated) circle onto a mask!
#It's also kinda not super useful for a multi-segmented detector (it will chop up your circle).


import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt

radius_size = 500
circle_center = ([100, 100])
pixel_size = 0.000075
print("\nI will now draw a circle with a radius of " + str(radius_size) + " centered at " + str(center) + ".\n")
print("\nOkidoki, script should be running now!\n")

h5FileName = str(sys.argv[1])
f = h5py.File(h5FileName, "r")
data = f['data']
data = data['data']
xt = data.shape[0]
xt = np.arange(1, xt, 1)
xm = xt.copy()
yt = data.shape[1]
yt = np.arange(1, yt, 1)
ym = yt.copy()
x = xm/pixel_size
y = ym/pixel_size

mask = np.ones((data.shape[0], data.shape[1]))
mask = np.uint16(mask) 

f.close()

min_lim_x = circle_center[1] - radius_size
max_lim_x = circle_center[1] + radius_size
min_lim_y = circle_center[0] - radius_size
max_lim_y = circle_center[0] + radius_size

for i in range(min_lim_x, max_lim_x):
    for j in range(min_lim_y, max_lim_y):
        radius = np.sqrt(pow(xm[i] - center[1], 2) + pow(ym[j] - center[0], 2))
        if radius <= radius_size :
            mask[i, j] = 0


plt.imshow(mask, interpolation="nearest", norm=None)
plt.show()

h5FileName = "circle-mask.h5"
print("writing file circle-mask.h5 ...\n")
f = h5py.File(h5FileName, "w") 
data = f.create_group("data")
data.create_dataset("data", data=mask)
f.close()

#h5FileName = input("\nIf you'd like to display the mask superimposed on an image, input a single frame file name:  \n")
#f = h5py.File(h5FileName, "r")
#data = f['data']
#data = data['data']
#tim = np.array(data[:, :])
#im = tim.copy()
#f.close()
                                                                                
#plt.imshow(1000*im*mask, interpolation="nearest", norm=None)
#plt.show()



