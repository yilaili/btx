import numpy as np
from scipy.signal import sosfiltfilt, butter

def radial_profile(data, center=None, mask=None, 
                   filter=False, filter_order=2, filter_threshold=0.25,
                   threshold=10):
    """
    Compute the radial intensity profile of an image.
    
    Parameters
    ----------
    data : numpy.ndarray, shape (n,m)
        detector image
    center : 2d tuple or array
        (cx,cy) detector center in pixels; if None, choose image center
    mask : numpy.ndarray, shape (n,m)
        detector mask, with zeros corresponding to pixels to mask
    filter : bool
        if True, apply a lowpass Butterworth filter to the radial intensity profile
    filter_order : int
        order of the Butterworth filter
    filter_threshold : float
        critical frequency for the bandpass Butterworth filter
    threshold : float
        below this intensity, set the radial intensity profile to zero
        
    Returns
    -------
    radialprofile : numpy.ndarray, 1d
        radial intensity profile of input image
    """
    y, x = np.indices((data.shape))
    if center is None:
        center = (int(data.shape[1]/2), int(data.shape[0]/2))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    if mask is not None:
        r = np.where(mask==1, r, 0)
    r = r.astype(np.int32)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr   = np.bincount(r.ravel())
    radialprofile = np.divide(tbin, nr, out=np.zeros(nr.shape[0]), where=nr!=0)
    if filter:
        sos = butter(filter_order, filter_threshold, output='sos')
        radialprofile = sosfiltfilt(sos, radialprofile)
    radialprofile[radialprofile<threshold] = 0
    return radialprofile

def pix2q(npixels, wavelength, distance, pixel_size):
    """
    Convert distance from number of pixels from detector center to q-space.
    
    Parameters
    ----------
    npixels : numpy.ndarray, 1d
        distance in number of pixels from detector center
    wavelength : float
        x-ray wavelength in Angstrom
    distance : float
        detector distance in mm
    pixel_size : float
        detector pixel size in mm
        
    Returns
    -------
    qvals : numpy.ndarray, 1d
        magnitude of q-vector in per Angstrom
    """
    theta = np.arctan(npixels*pixel_size/distance)
    return 2.*np.sin(theta/2.)/wavelength

def q2pix(qvals, wavelength, distance, pixel_size):
    """
    Convert distance from q-space to number of pixels from detector center.
    
    Parameters
    ----------
    qvals : numpy.ndarray, 1d
        magnitude of q-vector in per Angstrom
    wavelength : float
        x-ray wavelength in Angstrom
    distance : float
        detector distance in mm
    pixel_size : float
        detector pixel size in mm
        
    Returns
    -------
    npixels : numpy.ndarray, 1d
        distance in number of pixels from detector center
    """
    pix = np.arcsin(qvals*wavelength/2.)
    return 2*distance*pix/pixel_size
