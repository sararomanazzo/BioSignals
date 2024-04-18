#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 00:38:57 2023

@author: Marco Fronzi
"""

import cv2
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans



def eliminate_circular_shapes(image):
    # Convert image to grayscale if it's not already
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Threshold the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an all white image
    result_image = np.ones_like(gray) * 255
    
    for contour in contours:
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Compute the aspect ratio of the bounding rectangle
        aspect_ratio = float(w) / h
        
        # You may adjust the range below based on your requirement
        if 0.6 <= aspect_ratio <= 1.4:
            # This contour is too circular, skip it
            continue
        
        # Draw the contour on the result image
        cv2.drawContours(result_image, [contour], -1, (0, 0, 0), -1)  # -1 to fill the contour
    
    return result_image

def apply_kmeans(image, n_clusters=2, n_init=10):
    image_flat = image.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=0).fit(image_flat)
    clustered_image = kmeans.labels_.reshape(image.shape)
    return clustered_image


def apply_hough(image, method='line'):
    if method == 'line':
        # Detecting lines using Hough Line Transform
        lines = cv2.HoughLines(image, 1, np.pi / 180, 150)
        hough_image = np.zeros_like(image)
        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(hough_image, (x1, y1), (x2, y2), 255, 1)
        return hough_image
    else:
        # Add code for other types of Hough Transform (e.g., circle detection)
        pass

def apply_connected_components(image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    return num_labels, labels, stats, centroids



def mean_intensity(region):
    """Calculate mean intensity of a region"""
    return np.mean(region)

def std_intensity(region):
    """Calculate standard deviation of intensity of a region"""
    return np.std(region)

def area_function(image, mask):
    """
    Calculate the area of a specified region in an image.
    
    Parameters:
    - image: The image containing the region of interest.
    - mask: A binary mask representing the region of interest,
      where non-zero values indicate the region.
    
    Returns:
    - The area of the region.
    """
    # Ensure the mask is a boolean mask
    boolean_mask = mask > 0

    # Check that the dimensions of image and mask match
    if image.shape != mask.shape:
        raise ValueError(f"Dimensions of image {image.shape} and mask {mask.shape} do not match")

    # Now, boolean_mask should have the same dimensions as image,
    # so you can use it to index into image without causing an error.
    region = image[boolean_mask]

    # Count the number of True values in the boolean mask to calculate the area
    area = np.sum(boolean_mask)

    return area

def normalize_signal(img):
    """
    Normalize the image to the range [0, 255].
    """
    min_val = np.min(img)
    max_val = np.max(img)
    normalized_img = ((img - min_val) / (max_val - min_val)) * 255
    return normalized_img.astype(np.uint8)

def reduce_noise(img):
    """
    Reduce noise using Gaussian Blur.
    """
    return cv2.GaussianBlur(img, (5, 5), 0)



def area(region):
    """Calculate area of a region"""
    return np.sum(region > 0)  # Assuming binary mask

def aspect_ratio(region):
    """Calculate aspect ratio of a region"""
    rows, cols = np.where(region > 0)  # Assuming binary mask
    width = np.max(cols) - np.min(cols) + 1
    height = np.max(rows) - np.min(rows) + 1
    return height / width if width > 0 else 0

def perimeter(region):
    """Calculate perimeter of a region"""
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cv2.arcLength(contours[0], True) if contours else 0

def compactness(region):
    """Calculate compactness of a region"""
    perim = perimeter(region)
    ar = area(region)
    return (perim ** 2) / ar if ar > 0 else 0

def histogram_skewness(region):
    """Calculate skewness of the histogram of a region"""
    return skew(region.ravel())

def histogram_kurtosis(region):
    """Calculate kurtosis of the histogram of a region"""
    return kurtosis(region.ravel())

def entropy(region):
    """Calculate entropy of a region"""
    hist = cv2.calcHist([region], [0], None, [256], [0, 256])
    hist /= hist.sum()
    ent = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
    return ent

def extract_features(image_mask, original_image):
    # Use the mask to extract the region of interest from the original image
    region_of_interest = original_image[image_mask]
    
    # Now extract features from the region of interest
    features = {
        'Mean Intensity': mean_intensity(region_of_interest),
        'Standard Deviation of Intensity': std_intensity(region_of_interest),
        # 'Area': area_function(region_of_interest, image_mask),
        'Aspect Ratio': aspect_ratio(region_of_interest),
        'Perimeter': perimeter(region_of_interest),
        'Compactness': compactness(region_of_interest),
        'Histogram Skewness': histogram_skewness(region_of_interest),
        'Histogram Kurtosis': histogram_kurtosis(region_of_interest),
        'Entropy': entropy(region_of_interest)
    }
    
    return features


def write_features_to_file(item,features, file_path):
    with open(file_path, 'a') as file:
        for feature_name, feature_value in features.items():
            file.write(f"{item}, {feature_name}: {feature_value}\n")
            
            
            