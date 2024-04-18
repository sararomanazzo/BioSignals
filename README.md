# BioSignalProcessing

## Overview
This project is designed to provide a suite of image processing and data visualization tools, integrating functionalities from multiple Python scripts to offer a user-friendly graphical interface for processing images and generating data plots.

## Description
- **GUI.py**: This script provides the graphical user interface (GUI) for the application, allowing users to load images, initiate batch image processing, and visualize results. It utilizes Qt for managing the GUI components.
- **image_processing.py**: Contains core image processing functions such as noise reduction, normalization, and various analytical operations like calculating histogram data, and segmenting images into meaningful areas.
- **functions.py**: A utility module that includes a variety of image and data handling functions such as noise reduction, signal normalization, and feature extraction from image regions.
- **GUI_functions.py**: Supports the main GUI operations, including interactions like file dialogs and display updates.
- **plots.py**: Dedicated to generating and managing data plots that visualize various statistical properties of images.

## Features
- **Batch Image Processing**: Load and process multiple images simultaneously.
- **Histogram Analysis**: Generate and display histograms for processed image regions.
- **Image Feature Extraction**: Automatically extract and quantify features from images.
- **Data Visualization**: Plot data related to image properties and analysis results.

## Installation
To run this software, you'll need Python installed on your machine along with the following libraries:
- OpenCV
- NumPy
- Matplotlib
- PyQt5

You can install all required libraries using pip:
```bash
pip install numpy opencv-python matplotlib pyqt5
