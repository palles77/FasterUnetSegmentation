import cv2

def count_regions_and_pixels(image_path):
    # Load the binary image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}.")

    # Threshold the image to create a binary mask
    _, binary_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of regions (i.e., contours)
    num_regions = len(contours)

    # Count the number of white pixels in the image
    num_white_pixels = cv2.countNonZero(binary_mask)

    return num_regions, num_white_pixels
