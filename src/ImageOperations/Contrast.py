import cv2
import numpy as np
from skimage import restoration, exposure

"""
Attributes:
----------
None

Methods:
-------
apply_clahe(self, input_image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the input image.

apply_noise_filtering(self, input_image: np.ndarray, denoise_strength: float = 0.1, patch_size: int = 7, patch_distance: int = 21) -> np.ndarray:
    Applies Non-Local Means denoising to the input image.

apply_norm(self, input_image: np.ndarray, alpha: float, beta: float, norm_type: int) -> np.ndarray:
    Applies normalization to the input image using the specified parameters.

apply_median_filter(self, input_image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    Applies median filtering to the input image.

apply_normalization(self, input_image: np.ndarray, alpha: float = 1.0, beta: float = 0.0, norm_type: int = cv2.NORM_INF) -> np.ndarray:
    Applies normalization to the input image using the specified parameters and method.
"""
class Contrast():
    def __init__(self):
        """
        Initializes an instance of the ImageProcessor class.
        """
        None        

    #===============================================================#
    #                                                               #
    #===============================================================#    
    def apply_clahe(self, input_image: np.ndarray, clip_limit: float = 2.0, tile_grid_size = 8) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the input image.

        Parameters:
        ----------
        input_image: np.ndarray
            A 3D numpy array representing an input image.

        clip_limit: float, optional
            A float value representing the contrast limit for CLAHE. Default is 2.0.

        tile_grid_size: Tuple[int, int], optional
            A tuple of two integers representing the number of tiles in the horizontal and vertical directions
            used for CLAHE. Default is (8, 8).

        Returns:
        -------
        output_image: np.ndarray
            A 3D numpy array representing the processed output image.
        """
        hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        v_clahe = clahe.apply(v)
        hsv_clahe = cv2.merge((h, s, v_clahe))
        output_image = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)  # Store the processed image in a class attribute

        return output_image

    #===============================================================#
    #                                                               #
    #===============================================================#    
    def apply_noise_filtering(self, input_image: np.ndarray, denoise_strength: float = 0.1, patch_size: int = 7, patch_distance: int = 21) -> np.ndarray:
        """
        Applies Non-Local Means denoising to the input image.

        Parameters:
        ----------
        input_image: np.ndarray
            A 3D numpy array representing an input image.

        denoise_strength: float, optional
            A float value representing the strength of the denoising. Default is 0.1.

        patch_size: int, optional
            An integer representing the size of patches used for denoising
            Default is 7.

        patch_distance: int, optional
            An integer representing the distance between patches used for denoising. Default is 21.

        Returns:
        -------
        output_image: np.ndarray
            A 3D numpy array representing the processed output image.
        """
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        
        # Extract the value channel
        v_channel = hsv_image[:,:,2]
        
        # Apply Non-Local Means denoising to the value channel
        denoised_v_channel = restoration.denoise_nl_means(v_channel, patch_size=patch_size, patch_distance=patch_distance, h=denoise_strength, fast_mode=True)
        
        # Convert the denoised value channel to uint8 range
        denoised_v_channel_uint8 = np.round(denoised_v_channel * 255).astype(np.uint8)
        
        # Replace the value channel with the denoised value channel
        hsv_image[:,:,2] = denoised_v_channel_uint8
        
        # Convert the image back to RGB color space
        output_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        
        return output_image

    #===============================================================#
    #                                                               #
    #===============================================================#    
    def apply_median_filter(self, input_image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Applies median filtering to the input image.

        Parameters:
        ----------
        input_image: np.ndarray
            A 3D numpy array representing an input image.

        kernel_size: int, optional
            An integer representing the size of the kernel used for median filtering. Default is 3.

        Returns:
        -------
        output_image: np.ndarray
            A 3D numpy array representing the processed output image.
        """
        output_image = cv2.medianBlur(input_image, kernel_size)        
        return output_image

    #===============================================================#
    #                                                               #
    #===============================================================#
    def apply_normalization(self, input_image: np.ndarray, alpha: float = 1.0, beta: float = 0.0, norm_type: int = cv2.NORM_INF) -> np.ndarray:
        """
        Applies normalization to the input image using the specified parameters and method.

        Parameters:
        ----------
        input_image: np.ndarray
            A 3D numpy array representing an input image in BGR format.

        alpha: float, optional
            A float value representing the normalization factor. Default is 1.0.

        beta: float, optional
            A float value representing the normalization shift factor. Default is 0.0.

        norm_type: int, optional
            An integer representing the normalization method. Default is cv2.NORM_INF.

        Returns:
        -------
        output_image: np.ndarray
            A 3D numpy array representing the processed output image.
        """
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        
        # Extract the value channel
        v_channel = hsv_image[:,:,2]

        # Check the V channel range
        if np.min(v_channel) < 0 or np.max(v_channel) > 255:
            raise ValueError('V channel values are outside the range [0, 255]')

        # Choose the appropriate normalization method
        if norm_type == cv2.NORM_INF:
            norm_func = exposure.rescale_intensity
            norm_kwargs = {'in_range': 'image', 'out_range': (0, 255)}
        elif norm_type == cv2.NORM_L1:
            norm_func = exposure.equalize_adapthist
            norm_kwargs = {'clip_limit': alpha, 'nbins': beta, 'kernel_size': None}
        else:
            norm_func = exposure.equalize_hist
            norm_kwargs = {}

        # Apply the normalization to the V channel
        v_norm = norm_func(v_channel, **norm_kwargs)

        # Check the normalized V channel range
        if np.min(v_norm) < 0 or np.max(v_norm) > 255:
            raise ValueError('Normalized V channel values are outside the range [0, 255]')

        # Replace the value channel with the denoised value channel
        v_norm_uint8 = np.round(v_norm * 255).astype(np.uint8)
        hsv_image[:,:,2] = v_norm_uint8
        
        # Convert the image back to RGB color space
        output_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        
        return output_image
