import os
import cv2
import numpy as np
from functions import *
from plots import *
import sys
from matplotlib import *
from PyQt5.QtGui import QImage

# Load the image from the specified path
# path='/Users/marco/Dropbox/Work/WORKING_DIR_UNIMELB/ImageProcessing/'
# image_path = '/Users/marco/Dropbox/Work/WORKING_DIR_UNIMELB/ImageProcessing/3_Flare_software_development/TEST.jpg'
# # TEST_CONTROL
# image = cv2.imread(image_path)
# original_image = cv2.imread(image_path)
# # plt.imshow(original_image)
# # 
# main(original_image)

def main(original_image):
    # Basic checks and initial setup
    print('Entering processing function')
    # if not hasattr(original_image, 'shape'):
    #     print("Error: The input is not a valid image.")
    #     return None, None
    


    # original_image=original_image=cv2.imread(original_image)
    # Check if the image is grayscale or color
    if len(original_image.shape) == 2:
        # Image is already grayscale
        image_gray = original_image
        print('Orig Image is Gray ')
        print(original_image[0])
    elif len(original_image.shape) == 3 and original_image.shape[2] == 3:
        # Convert color image (BGR) to grayscale
        image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        print('Orig Image is clor scale')
    else:
        print("Error: Image format not recognized or unsupported")
        return None, None
    print('Processing Function')
    # image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(image_gray, (5, 5), 5)

    # Thresholding to create a binary image
    # _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print('binary')
    
    
    norm_image = normalize_signal(image_gray)
    blurred = cv2.GaussianBlur(norm_image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contour = eliminate_circular_shapes(binary)
    
    reference_mask = np.zeros_like(binary, dtype=bool)
    background_mask = np.ones_like(binary, dtype=bool)
    
    vertical_masks = []
    histograms = []
    vertical_data = {}
    reference_data = {} 
    background_data = {}
    signal_image = original_image
    
    signal_results = []
    ref_bg_results = []
    # Example masks you might have obtained from your image processing
    # signal1_mask, signal2_mask, reference_mask, background_mask = ... # your code to obtain these masks
    
 

    num_signals = 0  # Track the number of signals detected
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    for label in range(1, num_labels):  # Ignore the background label
        x, y, w, h, area = stats[label]
        aspect_ratio = h / w
        area_mask = (labels == label)
    
        if aspect_ratio > 1 and num_signals < 2:  # Limit to at most 2 signals
            num_signals += 1
            vertical_mask = np.zeros_like(binary, dtype=bool)
            vertical_mask[area_mask] = True
            vertical_masks.append(vertical_mask)
            cv2.rectangle(signal_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(signal_image, f'Signal {num_signals}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            vertical_data[f'Signal {num_signals}'] = extract_features(vertical_mask, original_image)
            signal_results.append(vertical_data[f'Signal {num_signals}'])
            # Generate histogram for the signal
            hist_signal = generate_histogram(original_image, vertical_mask)
            histograms.append(hist_signal)
        else:
            # Process non-signal areas
            reference_mask[area_mask] = True
            background_mask[area_mask] = False
    
    reference_data = extract_features(reference_mask, original_image)
    ref_bg_results.append(reference_data)
# After processing all labels
    x_ref, y_ref, w_ref, h_ref = cv2.boundingRect(reference_mask.astype(np.uint8))
    cv2.rectangle(signal_image, (x_ref, y_ref), (x_ref + w_ref, y_ref + h_ref), (0, 255, 0), 2)
    cv2.putText(signal_image, 'Control', (x_ref, y_ref - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    background_data = extract_features(background_mask, original_image)
    ref_bg_results.append(background_data)
    
# #   Generate histograms for reference and background areas
#     hist_reference = generate_histogram(original_image, reference_mask)
#     histograms.append(hist_reference)
#     hist_background = generate_histogram(original_image, background_mask)
#     histograms.append(hist_background)
    
    
    histograms = generate_histogram_data(original_image, vertical_masks, reference_mask, background_mask)


    # Corresponding labels for each area
    # areas = ["Signal 1", "Signal 2", "Reference", "Background"]
    
    # # List of masks (order should correspond to the labels)
    # masks = [signal1_mask, signal2_mask, reference_mask, background_mask]
    
    # # Create the results list with labels
    
    # for label, mask in zip(areas, masks):
    #     features = extract_features(mask, original_image)  # Assume this function returns a dictionary of features
    #     labeled_features = {"label": label}  # Add label to the features
    #     labeled_features.update(features)  # Include the extracted features
    #     results.append(labeled_features)

    # print(results)
    # Print the results for debug
    for key in vertical_data.keys():
        print(key, vertical_data[key])
    print('Reference:', reference_data)
    print('Background:', background_data)
    
    # plt.imshow(signal_image)
    # plt.imshow(original_image)
    return signal_image, signal_results, ref_bg_results,  histograms
    
    
    
    # norm_image=normalize_signal(image_gray)
    # blurred = cv2.GaussianBlur(norm_image, (5, 5), 0)
    # _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # contour = eliminate_circular_shapes(binary)
    # # print(contour)
    # # binary=normalize_signal(image_gray)
    # # rectangs=binary
    # # rectangs = eliminate_circular_shapes(binary)
    # # lines= apply_hough(binary)
    # # components=apply_connected_components(binary)
    # # print('original image type',type(original_image))
    # # plt.imshow(original_image)
    # # print(type(binary))
    # # plt.imshow(binary)
    # # print(type(rectangs))
    # # plt.imshow(rectangs)
    # # Initialize masks for reference and background areas
    # reference_mask = np.zeros_like(binary, dtype=bool)
    # background_mask = np.ones_like(binary, dtype=bool)  # Initially consider all pixels as background
    # reference_area = 0
    # background_area = 0
    
    # # Other initialization
    # vertical_masks = []
    # vertical_data = {}
    # reference_data = {} 
    # background_data = {}
    # background_area = np.sum(background_mask)
    # reference_area = np.sum(reference_mask)
    # vertical_area_counter = 0
    # signal_image=original_image
    
    # results=[]
     
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    # # Step 4: Analyze the Areas
    # for label in range(1, num_labels):  # Ignore the background label
    #     x, y, w, h, area = stats[label]
    #     aspect_ratio = h / w
    #     area_mask = (labels == label)
    #     print('label',label)
    #     if aspect_ratio > 1:  # Assuming vertical signals have a higher aspect ratio
    #         # Process vertical areas
            
    #         vertical_area_counter += 1
    #         vertical_mask = np.zeros_like(binary, dtype=bool)
    #         vertical_mask[area_mask] = True
    #         vertical_masks.append(vertical_mask)
    #         cv2.rectangle(signal_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #         cv2.putText(signal_image, f'Signal {vertical_area_counter}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    #         # vertical_intensity = np.mean(original_image[area_mask])
    #         vertical_data[f'Signal {vertical_area_counter}']  = extract_features(vertical_mask, original_image)
    #         print(f'Signal {vertical_area_counter}',vertical_data[f'Signal {vertical_area_counter}'] )
    #         results.append(vertical_data[f'Signal {vertical_area_counter}'])
    #         # print(results)
    #         # vertical_std = np.std(image[area_mask])
    #         # vertical_data[f'Signal {vertical_area_counter}'] = {
    #         #     'Area': area,
    #         #     'Intensity': vertical_intensity,
    #         #     'Std': vertical_std
    #         # }
    #         # print(f'Signal {vertical_area_counter} - Area: {area}, Intensity: {vertical_intensity}, Std: {vertical_std}')
    #     else:
    #         # Process reference areas
    #         print('label ref',label)
    #         reference_mask[area_mask] = True
    #         background_mask[area_mask] = False
    #         reference_area += area
    #     # refernce= extract_features(reference_mask, original_image)
    #     # background = extract_features(background_mask, original_image)
    # reference_data=extract_features(reference_mask, original_image)
    # results.append(reference_data)
    # background_data=extract_features(background_mask, original_image)
    # results.append(background_data)
    # cv2.rectangle(signal_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # # plt.imshow(binary)
    # cv2.putText(signal_image, 'Reference', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # print(vertical_data[f'Signal 1'] )
    # print(vertical_data[f'Signal 2'] )
    # print('ref',reference_data)
    # print('background', background_data)
    # print('-------------------------')
    # print(type(results))
    # return signal_image, results
    

    # print(vertical_area_counter,nsitytensity )
    # print(features)
    # # print(' start')
    # # print(rectangs)
    # # # plt.show(rectangs)
    # # print(' end')
    # # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(rectangs)
    
    # # print(num_labels, labels, stats, centroids)
    # # print('start loop', num_labels)
    # # features = []
    # # for label in range(1, num_labels):
    # #     print('label',label)
    # #     x, y, w, h, area = stats[label]
    # #     print('x y w h ',x, y, w, h, area)
    # #     aspect_ratio = h / w
    # #     area_mask = (labels == label)
    # #     print(area_mask)
    # #     # Create a mask for the current rectangle
    # #     # mask = np.zeros_like(image_gray)
    # #     # mask[y:y+h, x:x+w] = 1
    
    # #     # Extract features using the mask
    # #     # feature = extract_features(mask, original_image)
    # #     # features.append(feature)
    # #     print(aspect_ratio)
    # # clustered_image = apply_kmeans(binary, n_clusters=3, n_init=10)
    # sys.exit(-1)
    # # plt.imshow(blurred)

    # # Image Analysis
    
    # image_area = np.sum(rectangs) / 255
    # image_intensity = np.mean(rectangs)
    # image_std = np.std(rectangs)
    # # image_area = np.sum(binary) / 255
    # # image_intensity = np.mean(blurred)
    # # image_std = np.std(blurred)

    # # # # Initialization for analysis
    # reference_mask = np.zeros_like(binary, dtype=bool)
    # background_mask = np.ones_like(binary, dtype=bool)
    # vertical_masks = []
    # vertical_data = {}
    # vertical_area_counter = 0
    # signal_data = {}
    # print('lab start')

    # # # # Labeling and analyzing areas
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(rectangs)
    # print('num lab', num_labels)
    # for label in range(1, num_labels):
    #    print('label', label)
       
    #    # Extract the stats for the current label
    #    x, y, w, h, area = stats[label]
    #    aspect_ratio = h / w
    #    area_mask = (labels == label)
    #    print('label',label)
    #    print('other',x, y, w, h, area )
    #    process_vertical_areas(area_mask, label, x, y, w, h, area, image_gray, vertical_masks, vertical_data, signal_data)

    #    # if aspect_ratio > 1:  # Vertical signals
    #    #     process_vertical_areas(area_mask, label, x, y, w, h, area, image_gray, vertical_masks, vertical_data, signal_data)
    #    # else:
    #    #     process_reference_areas(area_mask, reference_mask, background_mask)
    #        # process_reference_areas(area_mask, label, x, y, w, h, area, image_gray, vertical_masks, vertical_data, signal_data)
    #    # process_reference_areas(area_mask, reference_mask, background_mask)
    

    # # # Post-processing and extracting features
    # # process_and_extract_features(image_mask, original_image)
    # original_image_features, signal_1_features, signal_2_features, reference_features, background_features = process_and_extract_features(original_image, vertical_masks, reference_mask, background_mask)
    # # original_image_features, signal_1_features, signal_2_features, reference_features, background_features = process_and_extract_features(original_image, vertical_masks, reference_mask, background_mask)
    # # Output results
    # # results = format_results(signal_data)
    # results = format_results(signal_data, original_image_features, signal_1_features, signal_2_features, reference_features, background_features)    # display_and_save_results(original_image, results)
    # processed_image=binary
    # # plt.imshow(pocessed_image)



# def def process_vertical_areas(area_mask, label, x, y, w, h, area, image, vertical_masks, vertical_data, signal_data):
    # ... function implementation

# def process_vertical_areas(area_mask, label, stats, image, vertical_masks, vertical_data, signal_data):
def process_vertical_areas(area_mask, label, x, y, w, h, area, image, vertical_masks, vertical_data, signal_data):
    print('entering vertical area')

    # x, y, w, h, area = stats[label]
    vertical_mask = np.zeros_like(image, dtype=bool)
    vertical_mask[area_mask] = True
    vertical_masks.append(vertical_mask)

    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(image, f'Signal {label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    vertical_intensity = np.mean(image[area_mask])
    vertical_std = np.std(image[area_mask])
    vertical_data[f'Signal {label}'] = {
        'Area': area,
        'Intensity': vertical_intensity,
        'Std': vertical_std
    }
    signal_data[f'Signal {label}'] = vertical_data[f'Signal {label}']



# def process_reference_areas(area_mask, label, x, y, w, h, area, image, vertical_masks, vertical_data, signal_data):
#     # for label in range(1, num_labels):  # Ignore the background label
#         x, y, w, h, area = stats[label]
#         aspect_ratio = h / w
#         area_mask = (labels == label)
        
#         if aspect_ratio > 1:  # Assuming vertical signals have a higher aspect ratio
#             # Process vertical areas
#             vertical_area_counter += 1
#             vertical_mask = np.zeros_like(binary, dtype=bool)
#             vertical_mask[area_mask] = True
#             vertical_masks.append(vertical_mask)
#             background_mask[area_mask] = False  # Exclude vertical areas from background
            
#             cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             cv2.putText(image, f'Signal {vertical_area_counter}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#             vertical_intensity = np.mean(image[area_mask])
#             vertical_std = np.std(image[area_mask])
#             vertical_data[f'Signal {vertical_area_counter}'] = {
#                 'Area': area,
#                 'Intensity': vertical_intensity,
#                 'Std': vertical_std
#             }
#             signal_data[f'Signal {vertical_area_counter}'] = vertical_data[f'Signal {vertical_area_counter}']  # <-- Add this line
#             print(f'Signal {vertical_area_counter} - Area: {area}, Intensity: {vertical_intensity}, Std: {vertical_std}')
#             # print('Signal {vertical_area_counter} Features:', signal_+{vertical_area_counter}+_features)
#             # write_features_to_file(f'Signal {vertical_area_counter}', signal_features,path_res+'Info'+'_'+file_name)
#         else:
#             # Process reference areas
#             reference_mask[area_mask] = True
#             background_mask[area_mask] = False


def generate_histogram_data(image, vertical_masks, reference_mask, background_mask):
    histogram_data = []

    num_signals = len(vertical_masks)
    if num_signals >= 1:
        # Add histograms for signals
        for mask in vertical_masks:
            histogram = np.histogram(image[mask].ravel(), bins=256)
            histogram_data.append(histogram)

    # Add histograms for reference and background
    histogram_data.append(np.histogram(image[reference_mask].ravel(), bins=256))
    histogram_data.append(np.histogram(image[background_mask].ravel(), bins=256))

    return histogram_data


def process_reference_areas(area_mask, reference_mask, background_mask):
    print('entering ref area')
    reference_mask[area_mask] = True
    background_mask[area_mask] = False


def process_and_extract_features(image, vertical_masks, reference_mask, background_mask):
    # Extract features for the original image
    original_image_features = {
        'Area': np.sum(image > 0),  # Assuming binary mask
        'Intensity': np.mean(image),
        'Std': np.std(image)
    }

    # Extract features for the signals
    signal_1_features = extract_features(vertical_masks[0], image) if len(vertical_masks) > 0 else None
    signal_2_features = extract_features(vertical_masks[1], image) if len(vertical_masks) > 1 else None
    
    # Extract features for the reference and background areas
    reference_features = extract_features(reference_mask, image)
    background_features = extract_features(background_mask, image)

    return original_image_features, signal_1_features, signal_2_features, reference_features, background_features



def format_results(signal_data, original_image_features, signal_1_features, signal_2_features, reference_features, background_features):
    results = ""

    if original_image_features:
        results += "Original Image - Area: {}, Intensity: {}, Std: {}\n".format(
            # original_image_features.get('Area', {},
            original_image_features.get('Area', 'N/A'),
            original_image_features.get('Intensity', 'N/A'),
            original_image_features.get('Std', 'N/A')
        )

    for signal, data in signal_data.items():
        results += f"{signal} - Area: {data.get('Area', 'N/A')}, Intensity: {data.get('Intensity', 'N/A')}, Std: {data.get('Std', 'N/A')}\n"

    if signal_1_features:
        results += "Signal 1 - Area: {}, Intensity: {}, Std: {}\n".format(
            signal_1_features.get('Area', 'N/A'),
            signal_1_features.get('Intensity', 'N/A'),
            signal_1_features.get('Std', 'N/A')
        )

    if signal_2_features:
        results += "Signal 2 - Area: {}, Intensity: {}, Std: {}\n".format(
            signal_2_features.get('Area', 'N/A'),
            signal_2_features.get('Intensity', 'N/A'),
            signal_2_features.get('Std', 'N/A')
        )

    if reference_features:
        results += "Reference Areas - Area: {}, Intensity: {}, Std: {}\n".format(
            reference_features.get('Area', 'N/A'),
            reference_features.get('Intensity', 'N/A'),
            reference_features.get('Std', 'N/A')
        )

    if background_features:
        results += "Background - Area: {}, Intensity: {}, Std: {}\n".format(
            background_features.get('Area', 'N/A'),
            background_features.get('Intensity', 'N/A'),
            background_features.get('Std', 'N/A')
        )

    return results


def generate_histogram(image, mask):
    # Ensure the mask is single channel and of type uint8
    if len(mask.shape) != 2 or mask.dtype != np.uint8:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask
        mask = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    histogram = cv2.calcHist([masked_image], [0], None, [256], [0, 256])
    return histogram
    
# def generate_histogram(image, mask):
#     masked_image = cv2.bitwise_and(image, image, mask=mask)
#     histogram = cv2.calcHist([masked_image], [0], None, [256], [0, 256])
#     return histogram


def display_and_save_results(original_image, image, path_res, file_name, results):
    display_processed_image(original_image, path_res, file_name)
    display_images(original_image, image, path_res, file_name)
    plot_identified_signals(original_image, image, path_res, file_name)
    plot_masks(image, path_res, file_name)
    plot_histograms(image, path_res, file_name)
    plot_original_histogram(original_image, path_res, file_name)

    # Write results to a text file
    results_file_path = os.path.join(path_res, file_name + '_results.txt')
    with open(results_file_path, 'w') as f:
        f.write(results)

# im, res , hist = main(original_image)
# plt.imshow(im)

# print(res)
# plt.imshow(original_image)
# Uncomment for command-line use
# if __name__ == "__main__":
#     main()
