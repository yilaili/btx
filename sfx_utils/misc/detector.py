import numpy as np

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
