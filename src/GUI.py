#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:09:52 2024

@author: marco
"""
import image_processing 
import copy
import os
import sys
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.figure import Figure
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QPushButton, QVBoxLayout, QWidget

from GUI_functions import *
import os
from PyQt5.QtWidgets import QLabel

# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
# app = QApplication(sys.argv)

# from copy import deepcopy

# from matplotlib.figure import Figure # Import matplotlib figure object
# #from matplotlib.backends.backend import FigureCanvasQTAgg as FigureCanvas 
# import matplotlib # Import matplotlib to set backend option
# matplotlib.use('QT5Agg') # Ensure using PyQt5 backend

from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.processed_image = None
        self.initUI()
        


    #     # # Additional setup...
    #     # self.show()
    # def initUI(self):
    #     self.setGeometry(100, 100, 1200, 800)
    #     self.setWindowTitle("ImgMgk Signal Processor")
    #     self.grid = QGridLayout(self)
    
    #     # Setup for buttons
    #     self.setupButtons()
    
    #     # Setup for figure and canvas for matplotlib plots
    #     self.figure = plt.figure(figsize=(10, 8))
    #     self.canvas = FigureCanvas(self.figure)
    #     self.grid.addWidget(self.canvas, 4, 0, 3, 2)  # Adjust grid position and span
    
    #     # Initialize QLabel widgets for results
    #     self.signal_results_label = QLabel("Signal Results: ", self)
    #     self.ref_bg_results_label = QLabel("Reference and Background: ", self)
    #     self.result_label = QLabel(" ", self)
    
    #     # Add labels to the grid layout at the bottom
    #     self.grid.addWidget(self.signal_results_label, 7, 0)  # Position for signal results
    #     self.grid.addWidget(self.ref_bg_results_label, 7, 1)  # Position for reference and background
    #     # self.grid.addWidget(self.result_label, 9, 0, 3, 4)  # Spanning both columns for overall results
    #     self.grid.addWidget(self.result_label, 3, 0)
    #     self.show()

    def initUI(self):
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowTitle("ImgMgk Signal Processor")
        self.grid = QGridLayout(self)
    
        # Setup for buttons
        self.setupButtons()
    
        # Setup for the first figure and canvas for images
        self.figure_images = plt.figure(figsize=(6, 4))
        self.canvas_images = FigureCanvas(self.figure_images)
        self.grid.addWidget(self.canvas_images, 3, 0, 2, 2)  # Rows 0-2 for images
    
        # Setup for the second figure and canvas for histograms
        self.figure_histograms = plt.figure(figsize=(4, 4))
        self.canvas_histograms = FigureCanvas(self.figure_histograms)
        self.grid.addWidget(self.canvas_histograms, 5, 0, 2, 2)  # Rows 4-6 for histograms
    
        # Initialize QLabel widgets for results
        self.signal_results_label = QLabel("Signal Results: ", self)
        self.ref_bg_results_label = QLabel("Control and Background: ", self)
        # self.result_label = QLabel(" ", self)
    
        # Add labels to the grid layout at the bottom
        self.grid.addWidget(self.signal_results_label, 7, 0)  # Row 7 for signal results
        self.grid.addWidget(self.ref_bg_results_label, 7, 1)  # Row 7 for reference and background
        # self.grid.addWidget(self.result_label, 4, 0)  # Row 3 for overall results
    
        # Additional setup...
        self.show()

    def setupButtons(self):
        # Load Image Button
        btn_load = QPushButton("Load Image", self)
        btn_load.clicked.connect(self.load_image)
        self.grid.addWidget(btn_load, 0, 0)

        # Process Image Button
        btn_process = QPushButton("Process Imgage", self)
        btn_process.clicked.connect(self.process_image)
        self.grid.addWidget(btn_process, 0, 1)

        # Restore Original Button
        btn_restore = QPushButton("Reset", self)
        btn_restore.clicked.connect(self.restore)
        self.grid.addWidget(btn_restore, 1, 0)

        # Save Image Button
        btn_save = QPushButton("Save Image", self)
        btn_save.clicked.connect(self.save_image)
        self.grid.addWidget(btn_save, 1, 1)


        # Batch Process Button
        btn_batch_process = QPushButton("Batch Process Images", self)
        btn_batch_process.clicked.connect(self.batch_process_images)
        self.grid.addWidget(btn_batch_process, 2, 0)
        
                # Quit Button
        btn_quit = QPushButton("Quit", self)
        btn_quit.clicked.connect(self.close_app)
        self.grid.addWidget(btn_quit, 2, 1)
        
        


        
        
    def process_image(self):
        self.hsv_image = np.copy(self.hsv_image) 
        self.processed_image, signal_results, ref_bg_results, histograms = image_processing.main(self.original_image)
        # plt.imshow(self.original_image)
        # Clear previous figures
        self.figure_images.clear()
        print(type(histograms))
        print(type(self.processed_image))
    
        # Displaying original and processed images in a 2x1 layout
        ax1 = self.figure_images.add_subplot(121)
        ax1.imshow(self.original_image_copy)
        ax1.set_title("Original Image", fontsize=4)  # Decreased title font size
        ax1.axis('off')
       
    
        ax2 = self.figure_images.add_subplot(122)
        ax2.imshow(self.processed_image)
        ax2.set_title("Processed Image", fontsize=4)  # Decreased title font size
        ax2.axis('off')
        # self.displayImages()
        # Display histograms and update canvas
        self.displayHistograms(histograms)
        # self.canvas.draw()
        self.displayResults(signal_results, ref_bg_results)
        # self.processed_image=None
        # self.histograms=[None]


    # def process_image(self):

    #     self.hsv_image = np.copy(self.hsv_image) 
    #     # self.processed_image=processed_image
           

    #     self.processed_image, signal_results, ref_bg_results,  histograms = image_processing.main(self.original_image)
    #     print("Histograms:", histograms)  # Add this line to check histograms content
    #     # print(results['Signal 1'])
    #     # self.figure.clear()  # Clear the figure before updating
        

    #     # Displaying original and processed images in a 2x1 layout
    #     ax2 = self.figure.add_subplot(322)
    #     ax2.set_title("Processed Image")
    #     plt.imshow(self.processed_image,)
    #     ax2.axis('off')

    #     # ax2 = self.figure.add_subplot(122)
    #     # ax2.imshow(processed_image)
    #     # ax2.set_title("Processed Image")
    #     # ax2.axis('off')
    #     # print(histograms)
    #     self.displayHistograms(histograms)
    #     # self.displayHistograms(histograms)
    #     # histograms = image_processing.generate_histogram_data(self.original_image, vertical_masks, reference_mask, background_mask)

    #     # Clear and update the canvas with new histograms
    #     # self.figure.clear()
    # # # Define a list of colors for the histograms
    # #     colors = ['blue', 'green', 'red', 'purple']

    # # # Clear and update the canvas with new histograms
    # #     # self.figure.clear()
    # #     for i, histogram in enumerate(histograms):
    # #         ax = self.figure.add_subplot(3, 2, i + 3)
    
    # #         # Plot histogram with specified color
    # #         ax.bar(histogram[1][:-1], histogram[0], width=2, color=colors[i % len(colors)])
    
    # #         # Set title and labels
    # #         ax.set_title(f"His {i + 1}", fontsize=4)
    
    # #         # Set tick parameters
    # #         ax.tick_params(axis='both', which='major', labelsize=2)
    
    # #         # Optional: Set x and y labels
    # #         ax.set_xlabel('Pixel Value', fontsize=2)
    # #         ax.set_ylabel('Freq', fontsize=2)
    
    #     self.canvas.draw()
    #     self.displayResults(signal_results, ref_bg_results)
        





    def load_image(self):

        #popup the file explorere box, extract th file name
        filename = QFileDialog.getOpenFileName(self,'select')
        # print(filename[0])

        #read the selected file using mpimg
        self.original_image = mpimg.imread(str(filename[0]),1)
        self.original_image_copy = np.copy(self.original_image)
        # self.original_image = cv2.resize(self.original_image,(256,256))
        # #convert the image to HSV, we will make changes only in the v channel of teh image
        self.hsv_image = cv2.cvtColor(self.original_image,cv2.COLOR_BGR2HSV)
        self.hsv_image_orig = np.copy(self.hsv_image)
        # #extract the v_channel of the image for applying the transformations
        self.v_channel = self.hsv_image[:,:,2]
        # #store the original image in another varibale, we may need in future to revert abck to original image
        self.v_channel_orig = np.copy(self.v_channel)
        
        
        # print(filename[0])

    #     #show the loaded image in a subplot
        plt.clf()
        self.processed_image=None
        self.histograms=[None]
        self.displayImages()
        # ax1 = self.figure_images.add_subplot(1,2,1)
        # #give a title to the image
        # ax1.set_title("Original Image", fontsize=4)
        # plt.imshow(self.original_image_copy)
        # ax1.axis('off')
        self.figure_images.tight_layout()
        self.canvas_images.draw()
        


    # def displayImages(self):
    #     # Clear previous figures
    #     # self.figure.clear()
    #     self.original_image = mpimg.imread(str(filename[0]),1)
    #     # Original Image
    #     ax1 = self.figure.add_subplot(321)
    #     ax1.imshow(self.original_image)
    #     ax1.set_title("Original Image")
    #     ax1.axis('off')

    #     # Processed Image
    #     ax2 = self.figure.add_subplot(322)
    #     ax2.imshow(self.processed_image)
    #     ax2.set_title("Processed Image")
    #     ax2.axis('off')

        # self.canvas.draw()
        
    def displayImages(self):
        # Clear previous figures
        self.figure_images.clear()
        if self.original_image is not None:
            # Display Original Image
            ax1 = self.figure_images.add_subplot(1, 2, 1)  # Smaller subplot size
            ax1.imshow(self.original_image)
            ax1.set_title("Original Image", fontsize=4)
            ax1.axis('off')
    
        if self.processed_image is not None:
            # Display Processed Image
            ax2 = self.figure_images.add_subplot(1, 2, 2)  # Smaller subplot size
            ax2.imshow(self.processed_image)
            ax2.set_title("Processed Image", fontsize=4)
            ax2.axis('off')
        else:
            None
        self.figure_images.tight_layout()
        self.canvas_images.draw()
        
    # def displayImages(self):
    #     # Clear previous figures
    #     self.figure_images.clear()
    #     # Check if original image exists and is not None
    #     if self.original_image is not None:
    #         # Display Original Image
    #         ax1 = self.figure_images.add_subplot(2, 2, 1)  # Smaller subplot size
    #         ax1.imshow(self.original_image)
    #         ax1.set_title("Original Image", fontsize=4)
    #         ax1.axis('off')
    #     else:
    #         print("No original image to display.")
    #     # Display Original Image
    #     # ax1 = self.figure.add_subplot(3, 2, 1)  # Smaller subplot size
    #     # ax1.imshow(self.original_image)
    #     # ax1.set_title("Original Image", fontsize=4)
    #     # ax1.axis('off')
    #     if self.original_image is not None:
    #         # Display Original Image
    #         x2 = self.figure_images.add_subplot(2, 2, 2)  # Smaller subplot size
    #         ax2.imshow(self.processed_image)
    #         ax2.set_title("Processed Image",  fontsize=4)
    #         ax2.axis('off')
    #     else:
    #         print("No processed image to display.")
    #     # Display Processed Image
    #     # ax2 = self.figure.add_subplot(3, 2, 2)  # Smaller subplot size
    #     # ax2.imshow(self.processed_image)
    #     # ax2.set_title("Processed Image",  fontsize=4)
    #     # ax2.axis('off')
    #     self.canvas_images.draw()
    
    def displayHistograms(self, histograms):
        self.figure_histograms.clear()
        colors = ['blue', 'green', 'red', 'purple']
        edge_colors = ['darkblue', 'darkgreen', 'darkred', 'indigo']
    
        # Determine the labels based on the number of histograms
        num_histograms = len(histograms)
        if num_histograms == 4:
            # labels = ["Background", "Control", "Signal 2", "Signal 1"]
            labels = ["Signal 1", "Signal 2", "Control", "Background"]
        elif num_histograms == 3:
            labels = ["Signal ", "Control", "Background"]
        elif num_histograms == 2:
            labels = ["Control", "Background"]
        else:
            labels = []
    
        # Ensure there are always four histogram spaces
        histograms = (histograms + [None] * 4)[:4]
    
        for i in range(4):
            ax = self.figure_histograms.add_subplot(2, 2, 4-i)
            histogram = histograms[i]
    
            if i < len(labels):
                label = labels[i]
                if histogram is not None:
                    ax.bar(histogram[1][:-1], histogram[0], width=2, color=colors[i % len(colors)], edgecolor=edge_colors[i % len(edge_colors)])
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                    ax.set_xlim(0, 256)
                    ax.axis('off')
                ax.set_title(label, fontsize=2)
            else:
                ax.axis('off')
    
            ax.tick_params(axis='both', which='major', labelsize=0)
            ax.set_xlabel('Pixel Value', fontsize=2)
            ax.set_ylabel('Freq', fontsize=2)
            ax.axis('off')
    
        self.figure_histograms.tight_layout()
        self.canvas_histograms.draw()
        


    #     self.canvas.draw()
    # def displayHistograms(self, histograms):
    #     colors = ['blue', 'green', 'red', 'purple']
    #     for i, histogram in enumerate(histograms):
    #         ax = self.figure.add_subplot(3, 2, i + 3)
    #         ax.bar(histogram[1][:-1], histogram[0], width=2, color=colors[i % len(colors)])
    #         ax.set_title(f"Histogram {i + 1}", fontsize=8)
    #         ax.tick_params(axis='both', which='major', labelsize=6)
    #         ax.set_xlabel('Pixel Value', fontsize=8)
    #         ax.set_ylabel('Frequency', fontsize=8)
    #         ax.axis('off')
    #     self.figure.tight_layout()
    #     self.canvas.draw() 
        
    # def displayHistograms(self, histograms):
    #     colors = ['blue', 'green', 'red', 'purple']
    #     edge_colors = ['darkblue', 'darkgreen', 'darkred', 'indigo']
    #     labels = ["Background", "Control", "Signal 2", "Signal 1"]
        
    #     # Ensure there are always four histograms (even if empty)
    #     histograms = (histograms + [None] * 4)[:4]
        
    #     for i in range(4):
    #         ax = self.figure.add_subplot(3, 2, 4-i)  # Count from last to first
    #         histogram = histograms[i]
    #         label = labels[i]
    #         if histogram is not None:
    #             ax.bar(histogram[1][:-1], histogram[0], width=8, color=colors[i % len(colors)], edgecolor=edge_colors[i % len(edge_colors)])
    #         else:
    #             ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
    #             ax.set_xlim(0, 256)
    #         ax.set_title(label, fontsize=2)
    #         ax.tick_params(axis='both', which='major', labelsize=0)
    #         ax.set_xlabel('Pixel Value', fontsize=2)
    #         ax.set_ylabel('Freq', fontsize=2)
    #         ax.axis('off')
    #     self.figure.tight_layout()
    #     self.canvas.draw()


    # def displayHistograms(self, histograms):
    #     colors = ['blue', 'green', 'red', 'purple']
    #     edge_colors = ['darkblue', 'darkgreen', 'darkred', 'indigo']
    #     # labels = ["Signal 1", "Signal 2", "Control", "Background"]
    #     labels = ["Background", "Control", "Signal 2", "Signal 1"]
    #     # Ensure there are always four histograms (even if empty)
    #     # histograms += [None] * (4 - len(histograms))
        
    #     for i, histogram in enumerate(histograms):
    #         ax = self.figure.add_subplot(3, 2, 4 - i)
    #         if histogram is not None:
    #             ax.bar(histogram[1][:-1], histogram[0], width=2, color=colors[i % len(colors)])
    #         else:
    #             ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
    #             ax.set_xlim(0, 256)
    #         ax.set_title(labels[i], fontsize=2)
    #         ax.tick_params(axis='both', which='major', labelsize=0)
    #         ax.set_xlabel('Pixel Value', fontsize=2)
    #         ax.set_ylabel('Freq', fontsize=2)
    #         ax.axis('off')
    #     self.figure.tight_layout()
    #     self.canvas.draw()
        
        
    # def displayHistograms(self, histograms):
    #     colors = ['blue', 'green', 'red', 'purple']
    #     default_labels = ["Signal 1", "Signal 2", "Reference", "Background"]
    
    #     for i, histogram in enumerate(histograms):
    #         ax = self.figure.add_subplot(4, 1, i + 1)
    #         label = default_labels[i] if i < len(histograms) else "No Data"
    #         if histogram is not None:
    #             ax.bar(histogram[1][:-1], histogram[0], width=2, color=colors[i % len(colors)])
    #         else:
    #             ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
    #             ax.set_xlim(0, 256)
    #         ax.set_title(label, fontsize=8)
    #         ax.tick_params(axis='both', which='major', labelsize=6)
    #         ax.set_xlabel('Pixel Value', fontsize=8)
    #         ax.set_ylabel('Frequency', fontsize=8)
    
    #     self.figure.tight_layout()
    #     self.canvas.draw()

        
        # for i, histogram in enumerate(histograms):
        #     ax = self.figure.add_subplot(3, 2, i + 3)
        #     ax.bar(histogram[1][:-1], histogram[0], width=0.5, color=colors[i % len(colors)], edgecolor=edge_colors[i % len(edge_colors)])
        #     ax.set_title(f"Hist {i + 1}", fontsize=4)
        #     ax.tick_params(axis='both', which='major', labelsize=8)
        #     ax.set_xlabel('Pixel Value', fontsize=2)
        #     ax.set_ylabel('Freq', fontsize=6)
        #     ax.axis('off')
        #     # plt.tight_layout()
        # self.figure.tight_layout()
        
        # self.canvas.draw()



    def displayResults(self, signal_results, ref_bg_results):
        signal_results_text = "Signal Results:\n"
        for result in signal_results:
            for key, value in result.items():
                formatted_value = format(value, '.4f') if isinstance(value, float) else str(value)
                signal_results_text += f"  {key}: {formatted_value}\n"
            signal_results_text += "\n"
    
        ref_bg_results_text = "Control and Background: \n"
        for result in ref_bg_results:
            if isinstance(result, dict):
                for key, value in result.items():
                    formatted_value = format(value, '.4f') if isinstance(value, float) else str(value)
                    ref_bg_results_text += f"  {key}: {formatted_value}\n"
            elif isinstance(result, (float, np.float64)):
                # Handle direct numeric values
                formatted_value = format(result, '.4f')
                ref_bg_results_text += f"Value: {formatted_value}\n"
            else:
                # Fallback for other types
                ref_bg_results_text += f"{result}\n"
            ref_bg_results_text += "\n"
    
        self.signal_results_label.setText(signal_results_text)
        self.ref_bg_results_label.setText(ref_bg_results_text)





    def restore(self):
        if hasattr(self, 'original_image'):
            
            # Reset the processed image to the original one
            self.original_image = None #self.original_image.copy()
            self.processed_image = None
            hist_reset = [None]
            ref_bg_results =[]
            signal_results = []

            # Update the display to show the original image
            self.displayImages()
            self.displayHistograms(hist_reset)

            self.displayResults(signal_results, ref_bg_results)
            
        else:
            print("No image to reset.")

 



    def save_image(self):
    # Check if processed image is available
        if hasattr(self, 'processed_image'): # and self.processed_image is not None:
            self.save_processed_image()
            self.save_histograms_and_results()
        else:
            print("No processed image to save or image is empty.")

    def save_processed_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Processed Image", "",
                                                  "Images (*.png *.xpm *.jpg);;All Files (*)")#, options=options)
        if fileName:
            cv2.imwrite(fileName, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))

    def save_histograms_and_results(self):
        # Save Histograms
        histogram_save_path, _ = QFileDialog.getSaveFileName(self, "Save Histograms", "",
                                                            "Images (*.png *.xpm *.jpg);;All Files (*)")#, options=options)
        if histogram_save_path:
            self.figure_histograms.savefig(histogram_save_path)

        # Save Results
        results_save_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "",
                                                          "Text Files (*.txt);;All Files (*)")#, options=options)
        if results_save_path:
            with open(results_save_path, 'w') as file:
                file.write(self.signal_results_label.text() + "\n" + self.ref_bg_results_label.text())


    def batch_process_images(self):
        self.load_multiple_images()
        if not self.images:
            print("No valid images to process.")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not save_dir:
            print("No directory selected for saving results.")
            return

        for index, image in enumerate(self.images):
            s_image, sig_res, ref_bg_res, hist = image_processing.main(image)

            # Save the processed image
            image_path = os.path.join(save_dir, f"processed_image_{index}.png")
            cv2.imwrite(image_path, cv2.cvtColor(s_image, cv2.COLOR_RGB2BGR))

            # Save histograms and results
            self.save_histograms(hist, save_dir, index)
            self.save_results(sig_res, ref_bg_res, save_dir, index)


    # def load_multiple_images(self):
    #     options = QFileDialog.Options()
    #     options |= QFileDialog.DontUseNativeDialog
    #     files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", 
    #                                             "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
    #     self.images = [cv2.imread(file) for file in files if cv2.imread(file) is not None]
        
    def load_multiple_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", 
                                                "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if not files:
            print("No images selected.")
            return

        self.images = []
        for file in files:
            img = cv2.imread(file)
            if img is not None:
                self.images.append(img)
            else:
                print(f"Warning: Unable to load image at {file}")

    # def batch_process_images(self):
    #     self.load_multiple_images()
    #     if not self.images:
    #         print("No valid images to process.")
    #         return
    #     processed_images = [image_processing.main(image)[0] for image in self.images]
    #     self.batch_save_images(processed_images)


    def batch_save_images(self, processed_images):
        save_directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        for i, image in enumerate(processed_images):
            cv2.imwrite(f"{save_directory}/processed_image_{i}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # Additional code to save histograms and data

        print("Batch processing and saving completed.")
        
    def batch_process_images(self):
        self.load_multiple_images()
        if not self.images:
            print("No valid images to process.")
            return

        # Directory for saving results
        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not save_dir:
            print("No directory selected for saving results.")
            return

        for index, image in enumerate(self.images):
            s_image, sig_res, ref_bg_res,  histograms= image_processing.main(image)

            # Save the processed image
            image_path = os.path.join(save_dir, f"processed_image_{index}.png")
            cv2.imwrite(image_path, cv2.cvtColor(s_image, cv2.COLOR_RGB2BGR))

            # Save histograms
            self.save_histograms(histograms, save_dir, index)

            # Save results
            self.save_results(sig_res, ref_bg_res, save_dir, index)

    def save_histograms(self, histograms, save_dir, index):
        for hist_index, histograms in enumerate(histograms):
            plt.figure()
            plt.bar(histograms[1][:-1], histograms[0], width=2)
            plt.title(f"Histogram {hist_index + 1}")
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            histogram_path = os.path.join(save_dir, f"histogram_{index}_{hist_index}.png")
            plt.savefig(histogram_path)
            plt.close()

    def save_results(self, sig_res, ref_bg_res, save_dir, index):
        # For signal results
        signal_results_text = "\n".join(f"{key}: {format(value, '.4f')}" 
                                        for result in sig_res 
                                        for key, value in result.items())
        signal_results_path = os.path.join(save_dir, f"signal_results_{index}.txt")
        with open(signal_results_path, "w") as file:
            file.write(signal_results_text)
    
        # For reference and background results
        ref_bg_results_text = "\n".join(f"{key}: {format(value, '.4f')}" 
                                        for result in ref_bg_res 
                                        for key, value in result.items())
        ref_bg_results_path = os.path.join(save_dir, f"ref_bg_results_{index}.txt")
        with open(ref_bg_results_path, "w") as file:
            file.write(ref_bg_results_text)
            
            
    def close_app(self):
        self.close()

def run():
    app = QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run()

