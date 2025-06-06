import sys
sys.path.append('./') if './' not in sys.path else None
sys.path.append('../') if '../' not in sys.path else None

from PIL import Image
import numpy as np
import scipy.ndimage as ndimage
from skimage.measure import regionprops
import cv2

#--------------------------------------------------------#
#                                                        #
#--------------------------------------------------------#
def eccentricity_filter(regions, threshold):
    """
    Filter regions based on their eccentricity (ovalness).
    
    Args:
        regions (list): List of regions to filter.
        threshold (float, optional): Threshold value for eccentricity. Defaults to 0.7.
    
    Returns:
        list: Filtered list of regions.
    """
    return [r for r in regions if r.eccentricity < threshold and r.area > 100]

#--------------------------------------------------------#
#                                                        #
#--------------------------------------------------------#
def get_brightest_region_internal(regions, grey_image_file_name):
    """
    Get the largest region by area.
    
    Args:
        regions (list): List of regions to search.
        grey_image_file_name (str): File name of the grayscale image.
    
    Returns:
        object: Largest region object.
    """
    max_region = None
    max_region_grey_level = 0
    
    # read the image
    grey_image = cv2.imread(grey_image_file_name, cv2.IMREAD_GRAYSCALE)
    
    for region in regions:
        average_grey_level: float = 0.0
        for coord in region.coords:
             average_grey_level += grey_image[coord[0], coord[1]]
        average_grey_level /= region.area
        
        if average_grey_level > max_region_grey_level:
            max_region_grey_level = average_grey_level
            max_region = region
           
    return max_region, max_region_grey_level

#--------------------------------------------------------#
#                                                        #
#--------------------------------------------------------#
def get_brightest_region(grey_input_file_name: str, 
                         bw_input_file_name: str, 
                         eccentricity_level: float, 
                         is_region_oval: bool):
    """
    Calculates the ratio of white pixels to black pixels in the largest oval region of an image.
    
    Args:
        grey_input_file_name (str): File name of the grayscale image.
        bw_input_file_name (str): File name of the black and white image.
        eccentricity_level (float): Threshold value for eccentricity (ovalness).
        is_region_oval (bool): If True, the region must be oval. If False, the region can be any shape.
    
    Returns:
        tuple: Bounding box of the largest oval region, ratio of white pixels to black pixels. Plus average grey level of the brightest region.
    """
    # load image with PIL
    with Image.open(bw_input_file_name) as img_pil:
        img_pil = img_pil.convert('L')
        # Convert PIL image to numpy array
        img = np.array(img_pil)
        
    # get the connected components in the image
    label_img, _ = ndimage.label(img)

    # get properties of the connected components
    regions = regionprops(label_img)

    if is_region_oval:
        # get only regions with low eccentricity (more oval-like)
        regions = eccentricity_filter(regions, eccentricity_level)

    # get the largest oval region
    brightest_oval_region, brightest_average_level = get_brightest_region_internal(regions, grey_input_file_name)

    result_region = None
    result_level = 0.0
    if brightest_oval_region is not None:        
        result_level = brightest_average_level
        result_region = brightest_oval_region

    return result_region, result_level

#--------------------------------------------------------#
#                                                        #
#--------------------------------------------------------#
def save_brightest_region(region, input_image_file_name):
    """
    Given a region of interest, reads an input image, masks the region, and saves the masked image as output.

    Args:
        region: The region of interest.
        input_image_file_name: A string representing the input image file name.
    """
    
    # Read the input image as grayscale
    input_grey_image = cv2.imread(input_image_file_name, cv2.IMREAD_GRAYSCALE)
    output_image_file_name = input_image_file_name.replace('.png', '_single_region.png')

    # Extract the region of interest from the input image
    minr, minc, maxr, maxc = region.bbox
    mask = np.zeros_like(input_grey_image)
    for coord in region.coords:
        mask[coord[0], coord[1]] = 1
    masked_input_image = input_grey_image * mask

    # Save the masked image as output
    cv2.imwrite(output_image_file_name, masked_input_image)
    
    return output_image_file_name
