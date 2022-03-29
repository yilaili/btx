import numpy as np
import psana
from psana import DataSource
from psana import EventId
from PSCalib.GeometryAccess import GeometryAccess

class PsanaInterface:

    def __init__(self, exp, run, det_type, track_timestamps=False):
        self.exp = exp # experiment name, string
        self.run = run # run number, int
        self.det_type = det_type # detector name, string
        self.track_timestamps = track_timestamps # bool, keep event info
        self.seconds, self.nanoseconds, self.fiducials = [], [], []
        self.ds = psana.DataSource(f'exp={exp}:run={run}')
        self.det = psana.Detector(det_type, self.ds.env())
        
    def get_pixel_size(self):
        """
        Retrieve the detector's pixel size in millimeters.
        
        Returns
        -------
        pixel_size : float
            detector pixel size in mm
        """
        return self.det.pixel_size(self.ds.env()) / 1.0e3
    
    def get_wavelength(self):
        """
        Retrieve the detector's wavelength in Angstrom.
        
        Returns
        -------
        wavelength : float
            wavelength in Angstrom
        """
        return self.ds.env().epicsStore().value('SIOC:SYS0:ML00:AO192') * 10.
    
    def estimate_distance(self):
        """
        Retrieve an estimate of the detector distance in mm.
        
        Returns
        -------
        distance : float
            estimated detector distance
        """
        return -1*np.mean(self.det.coords_z(self.run))/1e3

    def get_timestamp(self, evtId):
        """
        Retrieve the timestamp (seconds, nanoseconds, fiducials) associated with the input 
        event and store in self variables. For further details, see the example here:
        https://confluence.slac.stanford.edu/display/PSDM/Jump+Quickly+to+Events+Using+Timestamps
        
        Parameters
        ----------
        evtId : psana.EventId
            the event ID associated with a particular image
        """
        self.seconds.append(evtId.time()[0])
        self.nanoseconds.append(evtId.time()[1])
        self.fiducials.append(evtId.fiducials())
        return
    
    def get_images(self, num_images, assemble=True):
        """
        Retrieve a fixed number of images from the run. If the pedestal or gain 
        information is unavailable and unassembled images are requested, return
        uncalibrated images.
        
        Paramters
        ---------
        num_images : int
            number of images to retrieve
        assemble : bool, default=True
            whether to assemble panels into image
            
        Returns
        -------
        images : numpy.ndarray, shape ((num_images,) + det_shape)
            images retrieved sequentially from run, optionally assembled
        """
        counter = 0
        calibrate = True

        if assemble:
            images = np.zeros((num_images, 
                               self.det.image_xaxis(self.run).shape[0], 
                               self.det.image_yaxis(self.run).shape[0]))
        else:
            images = np.zeros((num_images,) + self.det.shape())
        
        for num,evt in enumerate(self.ds.events()):
            if counter < num_images:
                # check that pedestal and gain information are available
                if counter == 0:
                    if (self.det.pedestals(evt) is None) or (self.det.gain(evt) is None):
                        calibrate = False
                        if not calibrate and not assemble:
                            print("Warning: calibration data unavailable, returning uncalibrated data")

                # retrieve image, by default calibrated and assembled into detector format
                if assemble:
                    if not calibrate:
                        raise IOError("Error: calibration data not found for this run.")
                    else:
                        images[counter] = self.det.image(evt=evt)
                else:
                    if calibrate:
                        images[counter] = self.det.calib(evt=evt)
                    else:
                        images[counter] = self.det.raw(evt=evt)

                # optionally store timestamps associated with images
                if self.track_timestamps:
                    self.get_timestamp(evt.get(EventId))

            else:
                break
            counter += 1

        if counter < num_images:
            print("It appears we have reached the end of the run")
            images = images[:counter]
            
        return images


#### Miscellaneous functions ####

def retrieve_pixel_index_map(geom):
    """
    Retrieve a pixel index map that specifies the relative arrangement of
    pixels on an LCLS detector.
    
    Parameters
    ----------
    geom : string or GeometryAccess Object
        if str, full path to a psana *-end.data file
        else, a PSCalib.GeometryAccess.GeometryAccess object
    
    Returns
    -------
    pixel_index_map : numpy.ndarray, 4d
        pixel coordinates, shape (n_panels, fs_panel_shape, ss_panel_shape, 2)
    """
    if type(geom) == str:
        geom = GeometryAccess(geom)

    temp_index = [np.asarray(t) for t in geom.get_pixel_coord_indexes()]
    pixel_index_map = np.zeros((np.array(temp_index).shape[2:]) + (2,))
    pixel_index_map[:,:,:,0] = temp_index[0][0]
    pixel_index_map[:,:,:,1] = temp_index[1][0]
    
    return pixel_index_map.astype(np.int64)

def assemble_image_stack_batch(image_stack, pixel_index_map):
    """
    Assemble the image stack to obtain a 2D pattern according to the index map.
    Either a batch or a single image can be provided. Modified from skopi.
    
    Parameters
    ----------
    image_stack : numpy.ndarray, 3d or 4d 
        stack of images, shape (n_images, n_panels, fs_panel_shape, ss_panel_shape)
        or (n_panels, fs_panel_shape, ss_panel_shape)
    pixel_index_map : numpy.ndarray, 4d
        pixel coordinates, shape (n_panels, fs_panel_shape, ss_panel_shape, 2)
    
    Returns
    -------
    images : numpy.ndarray, 3d
        stack of assembled images, shape (n_images, fs_panel_shape, ss_panel_shape)
        of shape (fs_panel_shape, ss_panel_shape) if ony one image provided
    """
    if len(image_stack.shape) == 3:
        image_stack = np.expand_dims(image_stack, 0)
    
    # get boundary
    index_max_x = np.max(pixel_index_map[:, :, :, 0]) + 1
    index_max_y = np.max(pixel_index_map[:, :, :, 1]) + 1
    # get stack number and panel number
    stack_num = image_stack.shape[0]
    panel_num = image_stack.shape[1]

    # set holder
    images = np.zeros((stack_num, index_max_x, index_max_y))

    # loop through the panels
    for l in range(panel_num):
        images[:, pixel_index_map[l, :, :, 0], pixel_index_map[l, :, :, 1]] = image_stack[:, l, :, :]
        
    if images.shape[0] == 1:
        images = images[0]

    return images

def disassemble_image_stack_batch(images, pixel_index_map):
    """
    Diassemble a series of 2D diffraction patterns into their consituent panels. 
    Function modified from skopi.
        
    Parameters
    ----------
    images : numpy.ndarray, 3d
        stack of assembled images, shape (n_images, fs_panel_shape, ss_panel_shape)
        of shape (fs_panel_shape, ss_panel_shape) if ony one image provided
    pixel_index_map : numpy.ndarray, 4d
        pixel coordinates, shape (n_panels, fs_panel_shape, ss_panel_shape, 2)

    Returns
    -------
    image_stack_batch : numpy.ndarray, 3d or 4d 
        stack of images, shape (n_images, n_panels, fs_panel_shape, ss_panel_shape)
        or (n_panels, fs_panel_shape, ss_panel_shape)
    """
    if len(images.shape) == 2:
        images = np.expand_dims(images, axis=0)

    image_stack_batch = np.zeros((images.shape[0],) + pixel_index_map.shape[:3])
    for panel in range(pixel_index_map.shape[0]):
        idx_map_1 = pixel_index_map[panel, :, :, 0]
        idx_map_2 = pixel_index_map[panel, :, :, 1]
        image_stack_batch[:,panel] = images[:,idx_map_1,idx_map_2]
        
    if image_stack_batch.shape[0] == 1:
        image_stack_batch = image_stack_batch[0]
        
    return image_stack_batch
