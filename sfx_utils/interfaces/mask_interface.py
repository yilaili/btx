import numpy as np
import h5py, sys
from sfx_utils.interfaces.psana_interface import *

class MaskInterface:
    
    def __init__(self, exp, run, det_type):
        self.exp = exp # experiment name, str
        self.run = run # run number, int
        self.det_type = det_type # detector name, str
        
    def generate_from_psana_run(self, thresholds, n_images=1):
        """
        Generate a mask by extracting the first num_images from the indicated run,
        thresholding each, and then merging them. 
        
        Parameters
        ----------
        thresholds : tuple, 2d
            (lower, upper) thresholds for pixel value
        n_images : int
            number of images to threshold
            
        Returns
        -------
        mask : numpy.ndarray, shape (n_panels, n_pixels_fs, n_pixels_ss)
            binary mask, where 0 indicates a bad pixel
        """
        # retrieve images from run
        psi = PsanaInterface(exp=self.exp, run=self.run, det_type=self.det_type) 
        self.geom = psi.det.geometry(self.run)
        images = psi.get_images(n_images, assemble=False)
        
        # apply thresholds and set valid pixels to 1
        images[(images < thresholds[0]) | (images > thresholds[1])] = 0
        mask = np.prod(images, axis=0)
        mask[mask!=0] = 1
        
        self.mask = mask.astype(int)
        return
    
    def _save_as_hdf5(self, output, mask):
        """
        Save input mask to hdf5 format, reshaped for CrystFEL compatibility.
        Reshaping is based on the looping performed here:
        https://github.com/cctbx/dxtbx/blob/main/src/dxtbx/format/FormatXTCEpix.py#L46-L57.
        
        Parameters
        ----------
        output : str
            path to output mask file
        mask : numpy.ndarray, shape (n_panels * n_pixels_fs, n_pixels_ss)
            binary mask 
        """
        f = h5py.File(output, "w")
        dset = f.create_dataset('/entry_1/data_1/mask', data=mask, dtype='int')
        f.close()
        return
    
    def _convert_to_cctbx(self, mask):
        """
        Remake mask in CCTBX format, by reshaping and converting to flex.bool.
        Currently only the Jungfrau and Epix10k detectors are supported.
        
        Parameters
        ----------
        mask : numpy.ndarray, shape (n_panels, n_pixels_fs, n_pixels_ss)
            binary mask, where 0 indicates a bad pixel
        
        Returns
        -------
        reshaped_mask : list of flex bool objects
            binary mask in CCTBX format
        """
        from scitbx.array_family import flex
        
        recognized_det = True
        if 'epix10k' in det_type:
            n_asics_per_module = 4 # 2x2 asics per module
        elif 'jungfrau' in det_type:
            n_asics_per_module = 8 # 2x4 asics per module
        else:
            recognized_det = False

        if not recognized_det:
            sys.exit("Sorry, detector type currently not supported for saving to CCTBX format")
        else:
            reshaped_mask = []
            sdim, fdim = int(mock.shape[1]/2), int(mock.shape[2]/2)
            for n_panel in range(mock.shape[0]):
                for n_asic in range(n_asics_per_module):
                    sensor_id = n_asic // 2
                    asic_in_sensor_id = n_asic % 2
                    asic_data = mock[n_panel][sensor_id * sdim : (sensor_id + 1) * sdim,
                                              asic_in_sensor_id * fdim : (asic_in_sensor_id + 1) * fdim,]
                    reshaped_mask.append(flex.double(np.array(asic_data)))

            return reshaped_mask
                
    def save_mask(self, output, mask_format='psana'):
        """
        Save mask to one of four formats: psana (.npy), psana_assembled (.npy), 
        crystfel (.h5), or cctbx (.pickle).
        
        Parameters
        ----------
        output : str
            path to output mask file
        mask_format : str
            output style - psana, psana_assembled, crystfel, or cctbx
        """
        if mask_format == 'psana':
            np.save(output, self.mask)
            
        elif mask_format == 'psana_assembled':
            pixel_index_map = retrieve_pixel_index_map(self.geom)
            np.save(output, assemble_image_stack_batch(self.mask, pixel_index_map))
        
        if mask_format == 'crystfel':
            mask = self.mask.reshape(-1, self.mask.shape[-1])
            self._save_as_hdf5(output, mask)

        if mask_format == 'cctbx':
            try:
                import easy_pickle
            except ImportError:
                sys.exit("Could not load libtbx library")
            
            # supposed to be a list of flex bool objects
            mask = self._convert_to_cctbx(self.mask)
            easy_pickle.dump(output, mask)

        return

#### Miscellaneous functions ####
            
def load_cctbx_mask(input_file, det_type):
    """
    Load a DIALS-generated mask for a detector composed of 2x2 asics panels.
    The data are reshaped from:
    (n_asics, fs_asics_shape, ss_asics_shape) to:
    (n_panels, fs_panel_shape, ss_panel_shape),
    where each panel is composed
    of 2x2 asics. 
    
    This unstacking reverses the stacking performed here:
    https://github.com/cctbx/dxtbx/blob/main/src/dxtbx/format/FormatXTCEpix.py#L46-L57
    
    Parameters
    ----------
    input_file : str
        path to dials/cctbx mask in pickle format
    det_type : str
        either 'epix10k2m' or 'jungfrau4M'
    
    Returns
    -------
    mask : numpy.ndarray, shape (n_panels, fs_panel_shape, ss_panel_shape)
        binary mask, reshaped from DIALS output
    """
    try:
        from libtbx import easy_pickle
    except ImportError:
        sys.exit("Could not load libtbx library")
    
    recognized_det = True
    if 'epix10k' in det_type:
        n_asics_per_module = 4 # 2x2 asics per module
    elif 'jungfrau' in det_type:
        n_asics_per_module = 8 # 2x4 asics per module
    else:
        recognized_det = False

    if not recognized_det:
        sys.exit("Sorry, detector type currently not supported for saving to CCTBX format")
    
    mask_reshape = easy_pickle.load(input_file)
    mask_reshape = np.array([m.as_numpy_array() for m in mask_reshape]).astype(int)

    # asic unstack into mmodules; there are 2x2 or 4x2 asics per module
    sdim, fdim = mask_reshape.shape[1:]
    mask = np.zeros((int(mask_reshape.shape[0]/n_asics_per_module), sdim*2, fdim*2))

    counter = 0
    for module_count in range(mask.shape[0]):
        for asic_count in range(4):
            sensor_id = asic_count // 2 
            asic_in_sensor_id = asic_count % 2 
            mask[module_count][sensor_id * sdim : (sensor_id + 1) * sdim,
                               asic_in_sensor_id * fdim : (asic_in_sensor_id + 1) * fdim,] =  mask_reshape[counter]
            counter+=1
            
    return mask
