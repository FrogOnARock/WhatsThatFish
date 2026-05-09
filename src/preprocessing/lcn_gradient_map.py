import numpy as np
import cv2


def local_contrast_normalization(
        image: bytes
):
    """
    Local contrast normalization calculation applied to get
    """

    # Convert streamed GCP image to np array
    arr = np.frombuffer(image, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_float = img_bgr.astype(np.float64)

    # Apply the Gaussian blur to get weighted average for local regions in image
    gaussian_image = cv2.GaussianBlur(img_float, (5,5), 0)

    # Subtract to get V
    image_subtract = np.subtract(img_float, gaussian_image)

    # Get standard deviation of image
    image_squared = np.square(image_subtract)
    image_gaus_squared = cv2.GaussianBlur(image_squared, (5, 5), 0)
    image_sqrt = np.sqrt(image_gaus_squared)

    # Scale V by std deviation of local group
    image_final = np.divide(image_subtract, np.maximum(image_sqrt, 1e-4))
    min_val = image_final.min()
    max_val = image_final.max()
    image_out = ((image_final - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    return image_out


def gradient_map(
    image: bytes = None
):
    """
    Implementing the Scharr gradient to compute image gradients in 3x3 Kernel.
    Will be used to provide further edge detection in images for object classification.
    Implemented by leveraging cv2.Scharr and then computing actual magnitude sqrt(gx**2 + gy**2)
    """

    with open("./images/100028747.jpeg", "rb") as f:
        image = f.read()

    image_bytes = np.frombuffer(image, dtype=np.uint8)
    img_gray = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)
    img_gray = img_gray.astype(np.float64)

    gx = cv2.Scharr(img_gray, ddepth=cv2.CV_64F, dx=1, dy=0)
    gy = cv2.Scharr(img_gray, ddepth=cv2.CV_64F, dx=0, dy=1)

    magnitude = np.sqrt(gx**2 + gy**2)
    min_val = magnitude.min()
    max_val = magnitude.max()
    image_out = ((magnitude - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    return image_out
