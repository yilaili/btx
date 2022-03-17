import numpy as np
from libtbx import easy_pickle

def load_from_dials_asics(input_file):
    """
    Load a DIALS-generated mask for a detector composed of 2x2 asics panels.
    This was developed based on an Epix10k mask; it remains to be verified
    that it works for the Jungfrau. The data are reshaped from:
    (n_asics, fs_asics_shape, ss_asics_shape) to:
    (n_panels, fs_panel_shape, ss_panel_shape), where each panel is composed
    of 2x2 asics. 
    
    This unstacking reverses the stacking performed here:
    https://github.com/cctbx/dxtbx/blob/main/src/dxtbx/format/FormatXTCEpix.py#L46-L57
    
    Parameters
    ----------
    input_file : string
        path to DIALS mask in npy or pickle format; the latter requires libtbx
    
    Returns
    -------
    mask : numpy.ndarray, shape (n_panels, fs_panel_shape, ss_panel_shape)
        binary mask, reshaped from DIALS output
    """
    if input_file.split(".")[-1] == 'npy':
        mask_reshape = np.load(input_file)
    elif input_file.split(".")[-1] == 'mask':
        try:
            from libtbx import easy_pickle
        except ImportError:
            print("Could not load libtbx library")
            sys.exit()
        mask_reshape = easy_pickle.load(input_file)
        mask_reshape = np.array([m.as_numpy_array() for m in mask_reshape]).astype(int)
    else:
        print("Mask must in either numpy or pickle format")

    # asic unstack into mmodules; there are 2x2 asics per module
    sdim, fdim = mask_reshape.shape[1:]
    mask = np.zeros((int(mask_reshape.shape[0]/4), sdim*2, fdim*2))

    counter = 0
    for module_count in range(mask.shape[0]):
        for asic_count in range(4):
            sensor_id = asic_count // 2 # 0 0 1 1 
            asic_in_sensor_id = asic_count % 2 # 0 1 0 1
            mask[module_count][sensor_id * sdim : (sensor_id + 1) * sdim,
                               asic_in_sensor_id * fdim : (asic_in_sensor_id + 1) * fdim,] =  mask_reshape[counter]
            counter+=1
            
    return mask
