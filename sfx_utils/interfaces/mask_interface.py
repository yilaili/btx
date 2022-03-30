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
    
    def _convert_to_cctbx(self):
        """
        Remake mask in CCTBX format, by reshaping and converting to flex.bool.
        Currently only the Jungfrau and Epix10k detectors are supported.
        
        Returns
        -------
        mask : list of flex bool objects
            binary mask in CCTBX format
        """
        from scitbx.array_family import flex
        
        if 'epix10k' in self.det_type:
            mask = unstack_asics(self.mask, self.det_type, dtype='double')
        
        elif 'jungfrau' in self.det_type:
            from xfel.util import jungfrau
            
            mask = []
            for n_panel in range(self.mask.shape[0]):
                mask.append(flex.bool(jungfrau.correct_panel(self.mask[n_panel].astype('float64'))))

        else:
            sys.exit("Sorry, detector type currently not supported for saving to CCTBX format")
            
        return mask
                
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
            print(f"Saved mask to {output} in CrystFEL format, with shape {mask.shape}")

        if mask_format == 'cctbx':
            try:
                from libtbx import easy_pickle
            except ImportError:
                sys.exit("Could not load libtbx library")
            
            # supposed to be a list of flex bool objects
            mask = self._convert_to_cctbx()
            easy_pickle.dump(output, mask)
            print(f"Saved mask to {output} in CCTBX format")

        return

#### Miscellaneous functions ####
    
def unstack_asics(image, det_type, dtype='double'):
    """
    Unstack the asics and convert to CCTBX data type - specifically, a list of 
    flex.double or flex.bool objects. For the epix10k2M, for example, the shape
    is updated from (16,352,384) -> (64,176,192). This mimics the code here:
    https://github.com/cctbx/dxtbx/blob/main/src/dxtbx/format/FormatXTCEpix.py#L46-L57.
    
    Parameters
    ----------
    image : numpy.ndarray, shape (n_panels, n_pixels_fs, n_pixels_ss)
        image in unassembled psana format
    det_type : str
        epix10k2M or jungfrau4M
    dtype : str
        return data type, double or (for masks) bool
    
    Returns
    -------
    reshaped_image : list of flex.bool or flex.double arrays
        list of unstacked asics data
    """
    try:
        from scitbx.array_family import flex
    except ImportError:
        sys.exit("Could not load libtbx flex library")
    
    n_asics_per_module = 0
    if 'epix10k' in det_type:
        n_asics_per_module = 4 # 2x2 asics per module
    if 'jungfrau' in det_type:
        n_asics_per_module = 8 # 2x4 asics per module
    if n_asics_per_module == 0:
        sys.exit("Sorry, detector type currently not supported for CCTBX-style asics-stacking.")
    
    reshaped_image = []
    sdim, fdim = int(image.shape[1]/2), int(image.shape[2]/2)
    for n_panel in range(image.shape[0]):
        for n_asic in range(n_asics_per_module):
            sensor_id = n_asic // int(n_asics_per_module/2)
            asic_in_sensor_id = n_asic % int(n_asics_per_module/2)
            asic_data = image[n_panel][sensor_id * sdim : (sensor_id + 1) * sdim,
                                      asic_in_sensor_id * fdim : (asic_in_sensor_id + 1) * fdim,]
            if dtype == 'double':
                reshaped_image.append(flex.double(np.array(asic_data)))
            elif dtype == 'bool':
                reshaped_image.append(flex.bool(np.array(asic_data)))
            else:
                sys.exit("Print unrecognized data type; must be bool or double")
            
    return reshaped_image

def load_cctbx_mask(input_file):
    """
    Load a CCTBX mask, converting from a list of flex.bools to 
    a numpy array.

    Parameters
    ----------
    input_file = str
        path to CCTBX mask in pickle format
    
    Returns
    -------
    mask : numpy.ndarray, CCTBX-compatible shape
        binary mask
    """
    try:
        from libtbx import easy_pickle
    except ImportError:
        sys.exit("Could not load libtbx flex library")

    mask = easy_pickle.load(input_file)
    mask = np.array([m.as_numpy_array() for m in mask])
    return mask
