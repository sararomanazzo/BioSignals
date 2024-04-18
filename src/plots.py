#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 00:39:23 2023

@author: Marco Fronzi
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np



# Usage:
# Assuming img is your image
# result_img = eliminate_circular_shapes(img)


def display_processed_image(original_image, path, file_name):
    plt.imshow(original_image, 'BrBG_r')
    plt.title('Processed Image with Signals Identified')
    plt.axis('off')
    plt.title('Identified Signals')
    plt.savefig(path + file_name + '_' + 'processed_image.png', dpi=300)
    plt.show()
    # plt.close()


def display_images(original_image, image_gray_2, blurred, binary, image_area, image_intensity, image_std, reduceNoise, path, file_name):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Create a figure with a 2x2 grid of sub-plots

    # Display the original image
    axs[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), cmap='BrBG_r')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    # Display the grayscale image
    axs[0, 1].imshow(image_gray_2, cmap='gray')
    axs[0, 1].set_title('Gray')
    axs[0, 1].axis('off')

    # Display the noise reduced image
    axs[1, 0].imshow(blurred, cmap='gray')
    axs[1, 0].set_title('Gray (Noise Reduced)' if reduceNoise else 'Gray')
    axs[1, 0].axis('off')

    # Display the binary image
    axs[1, 1].imshow(binary, cmap='gray')
    axs[1, 1].set_title('Binary')
    axs[1, 1].axis('off')

    plt.tight_layout()  # Adjusts spacing between sub-plots to minimize overlap
    plt.savefig(path + file_name + '_' + 'images.png', dpi=300)
    plt.show()
    # plt.close()



def plot_identified_signals(original_image, image, path, file_name):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Identified Signals')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(path + file_name + '_' + 'signals.png', dpi=300)
    plt.show()
    # plt.close()

# # Debugging: print shapes and sums of masks
# print(f'vertical_mask shape: {vertical_mask.shape}, sum: {np.sum(vertical_mask)}')
# print(f'reference_mask shape: {reference_mask.shape}, sum: {np.sum(reference_mask)}')
# print(f'background_mask shape: {background_mask.shape}, sum: {np.sum(background_mask)}')



def plot_masks(image, vertical_mask, reference_mask, background_mask, path, file_name):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Signals')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(vertical_mask, cmap='gray')
    plt.title('Signal 2')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(reference_mask, cmap='gray')
    plt.title('Reference')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(background_mask, cmap='gray')
    plt.title('Background')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(path + file_name + '_' + 'images.png', dpi=300)
    plt.show()
    # plt.close()



def plot_original_histogram(original_image, path, file_name):
    plt.figure(figsize=(8, 6))  # Create a new figure with specified size

    # Flatten the original image to get a 1D array of pixel values.
    # If the image is grayscale, you can use it directly. If it's colored, you might want to convert it to grayscale first.
    # Assuming `original_image` is in grayscale:
    pixel_values = original_image.ravel()

    # Create a histogram
    plt.hist(pixel_values, bins=256, color='blue', alpha=0.5)
    plt.title('Histogram of Original Image Pixel Values')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.tight_layout()  # Adjusts the spacing between subplots
    plt.savefig(path + file_name + '_' + 'original_histogram.png', dpi=300)  # Save the figure to a file
    plt.show()  # Display the figure
    # plt.close()

# Usage in main script or elsewhere:
# plot_original_histogram(original_image, path, file_name)


# def generate_histograms(image, vertical_masks, reference_mask, background_mask):
#     plt.figure(figsize=(12, 12))

#     num_signals = len(vertical_masks)
#     if num_signals == 2:
#         # Both signals exist
#         grid_rows = 3
#         grid_cols = 2
#         plt.subplot(3, 2, 1)
#         plt.hist(image[vertical_masks[0]].ravel(), bins=256, color='green', alpha=0.5, label='Signal 1')
#         plt.legend(loc='upper right')
#         plt.title('Histogram of Signal 1 Pixel Values')
#         plt.xlabel('Pixel Value')
#         plt.ylabel('Frequency')
        
#         plt.subplot(3, 2, 2)
#         plt.hist(image[vertical_masks[1]].ravel(), bins=256, color='purple', alpha=0.5, label='Signal 2')
#         plt.legend(loc='upper right')
#         plt.title('Histogram of Signal 2 Pixel Values')
#         plt.xlabel('Pixel Value')
#         plt.ylabel('Frequency')
        
#         plt.subplot(3, 2, 3)
#         plt.hist(image[reference_mask].ravel(), bins=256, color='blue', alpha=0.5, label='Reference')
#         plt.legend(loc='upper right')
#         plt.title('Histogram of Reference Pixel Values')
#         plt.xlabel('Pixel Value')
#         plt.ylabel('Frequency')
        
#         plt.subplot(3, 2, 4)
#         plt.hist(image[background_mask].ravel(), bins=256, color='red', alpha=0.5, label='Background')
#         plt.legend(loc='upper right')
#         plt.title('Histogram of Background Pixel Values')
#         plt.xlabel('Pixel Value')
#         plt.ylabel('Frequency')
        
#         plt.tight_layout()
#         # plt.savefig(path + file_name + '_' + 'hist.png', dpi=300)
#         plt.show()

#     elif num_signals == 1:
#         # Only one signal exists
#         grid_rows = 3
#         grid_cols = 1
#         plt.subplot(grid_rows, grid_cols, 1)
#         plt.hist(image[vertical_masks[0]].ravel(), bins=256, color='green', alpha=0.5, label='Signal 1')
#         plt.legend(loc='upper right')
#         plt.title('Histogram of Signal 1 Pixel Values')
#         plt.xlabel('Pixel Value')
#         plt.ylabel('Frequency')
        
#         plt.subplot(grid_rows, grid_cols, 2)
#         plt.hist(image[reference_mask].ravel(), bins=256, color='blue', alpha=0.5, label='Reference')
#         plt.legend(loc='upper right')
#         plt.title('Histogram of Reference Pixel Values')
#         plt.xlabel('Pixel Value')
#         plt.ylabel('Frequency')
        
#         plt.subplot(grid_rows, grid_cols, 3)
#         plt.hist(image[background_mask].ravel(), bins=256, color='red', alpha=0.5, label='Background')
#         plt.legend(loc='upper right')
#         plt.title('Histogram of Background Pixel Values')
#         plt.xlabel('Pixel Value')
#         plt.ylabel('Frequency')
        
#         plt.tight_layout()
#         # plt.savefig(path + '_' + 'hist.png', dpi=300)
#         plt.show()
#     else:
#         plt.subplot(2, 2, 1)
#         plt.hist(image[reference_mask].ravel(), bins=256, color='blue', alpha=0.5, label='Reference')
#         plt.legend(loc='upper right')
#         plt.title('Histogram of Reference Pixel Values')
#         plt.xlabel('Pixel Value')
#         plt.ylabel('Frequency')
        
#         plt.subplot(2, 2, 2)
#         plt.hist(image[background_mask].ravel(), bins=256, color='red', alpha=0.5, label='Background')
#         plt.legend(loc='upper right')
#         plt.title('Histogram of Background Pixel Values')
#         plt.xlabel('Pixel Value')
#         plt.ylabel('Frequency')
        
#         plt.tight_layout()
        
        

        # plt.savefig(path + file_name + '_' + 'hist.png', dpi=300)
        # plt.show()
        # plt.close()

def plot_histograms(image, vertical_masks, reference_mask, background_mask, path, file_name):
    plt.figure(figsize=(12, 12))

    num_signals = len(vertical_masks)
    if num_signals == 2:
        # Both signals exist
        grid_rows = 3
        grid_cols = 2
        plt.subplot(3, 2, 1)
        plt.hist(image[vertical_masks[0]].ravel(), bins=256, color='green', alpha=0.5, label='Signal 1')
        plt.legend(loc='upper right')
        plt.title('Histogram of Signal 1 Pixel Values')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        plt.subplot(3, 2, 2)
        plt.hist(image[vertical_masks[1]].ravel(), bins=256, color='purple', alpha=0.5, label='Signal 2')
        plt.legend(loc='upper right')
        plt.title('Histogram of Signal 2 Pixel Values')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        plt.subplot(3, 2, 3)
        plt.hist(image[reference_mask].ravel(), bins=256, color='blue', alpha=0.5, label='Reference')
        plt.legend(loc='upper right')
        plt.title('Histogram of Reference Pixel Values')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        plt.subplot(3, 2, 4)
        plt.hist(image[background_mask].ravel(), bins=256, color='red', alpha=0.5, label='Background')
        plt.legend(loc='upper right')
        plt.title('Histogram of Background Pixel Values')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(path + file_name + '_' + 'hist.png', dpi=300)
        plt.show()

    elif num_signals == 1:
        # Only one signal exists
        grid_rows = 3
        grid_cols = 1
        plt.subplot(grid_rows, grid_cols, 1)
        plt.hist(image[vertical_masks[0]].ravel(), bins=256, color='green', alpha=0.5, label='Signal 1')
        plt.legend(loc='upper right')
        plt.title('Histogram of Signal 1 Pixel Values')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        plt.subplot(grid_rows, grid_cols, 2)
        plt.hist(image[reference_mask].ravel(), bins=256, color='blue', alpha=0.5, label='Reference')
        plt.legend(loc='upper right')
        plt.title('Histogram of Reference Pixel Values')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        plt.subplot(grid_rows, grid_cols, 3)
        plt.hist(image[background_mask].ravel(), bins=256, color='red', alpha=0.5, label='Background')
        plt.legend(loc='upper right')
        plt.title('Histogram of Background Pixel Values')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(path + '_' + 'hist.png', dpi=300)
        plt.show()
    else:
        plt.subplot(2, 2, 1)
        plt.hist(image[reference_mask].ravel(), bins=256, color='blue', alpha=0.5, label='Reference')
        plt.legend(loc='upper right')
        plt.title('Histogram of Reference Pixel Values')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 2)
        plt.hist(image[background_mask].ravel(), bins=256, color='red', alpha=0.5, label='Background')
        plt.legend(loc='upper right')
        plt.title('Histogram of Background Pixel Values')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(path + file_name + '_' + 'hist.png', dpi=300)
        plt.show()
        # plt.close()

# Usage in main script or elsewhere:
# plot_histograms(image, vertical_masks, reference_mask, background_mask, path, file_name)



# def plot_histograms(image, vertical_masks, reference_mask, background_mask):
#     # ... (rest of the function logic here) ...

#     plt.tight_layout()
#     plt.show()

