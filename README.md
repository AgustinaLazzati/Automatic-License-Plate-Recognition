# 🚗 Automatic License Plate Recognition (ALPR)

This repository contains an algorithm that automatically **detects** and **recognizes** license plates from vehicle images.

**Automatic License Plate Recognition (ALPR)** is a computer vision task designed to **detect** and **read** vehicle license plates from images.  
It is widely used in **traffic monitoring**, **parking management**, and **security systems**.  

In this project, the focus is on **parking lot images**, where the goal is to design and compare multiple approaches for building a reliable system capable of detecting and recognizing license plates **accurately** and **consistently**.  

The main objectives include:
- Developing and evaluating **reproducible methods** for locating license plates and extracting their alphanumeric content.  
- Comparing **deep learning-based approaches** (e.g., YOLO models) with **traditional image processing techniques** to assess performance differences.  
- Understanding how each stage — from **preprocessing** to **recognition** — affects overall **accuracy**, **robustness**, and **generalization**.

This project explores **image processing**, **machine learning**, and **computer vision** strategies to analyze how different design decisions influence the effectiveness and reliability of a full ALPR pipeline.

Ultimately, this work addresses challenges related to:
- Handling **real-world image variations** (lighting, angle, and quality)
- Balancing **deep learning** and **traditional** methods
- Maintaining **consistent and reproducible** experimental setups  

---

## ALPR Pipeline

The ALPR system operates in **four main stages**:

### 1️ Data Collection  
Analysis of provided (`real_plates.zip`) and custom-acquired images to ensure consistent **lighting**, **color**, and **camera angles**.
Also, data augmentation was implemented.

### 2️ Detection  
License plates are identified using:
- **YOLOv5** and **YOLOv8** (COCO pre-trained)
- A **custom YOLOv8** model trained specifically for plate detection.

### 3️ Segmentation  
Plates are aligned and processed to extract individual characters using:
- **Contour** and **watershed** methods  
- **Preprocessing filters** for enhanced clarity

### 4️ Recognition  
Character recognition is achieved through:
- **HOG features** for descriptor extraction  
- **Two SVM classifiers** – one for digits and one for letters  
- Final reconstruction of full license plate strings

---

## 🧩 Code Structure

| Section | Description |
|----------|--------------|
| `DataExplorations` | Initial data exploration and visualization |
| `protocol` | Acquisition testing and definition |
| `YOLO` | YOLO model training and evaluation |
| `CharacterSegmentation` | Character segmentation pipeline |
| `CharacterDescriptors` | Feature extraction and classification |

---

##  Plots & Visuals

- General plots: `plots/`  
- YOLO test results: `runs/detect/`  
- HSV analysis (Fréchet + FFT): `plots/hsv_plots/`  
- Character validation plots: `CharacterDescriptors/`  

> 🧠 Over **100 plots** were generated during project development!

---

## Dataset Used

- **`real_plates.zip`** – Real-world vehicle images (front & side views)  
- **`example_fonts.zip`** – Synthetic Spanish license plate font dataset  
  - `digitsIms.pkl` / `alphabetIms.pkl`: Cropped digits and letters  
  - `digitsLabels` / `alphabetLabels`: Label files for each character  
- **Own Images** – Additional dataset collected under a standardized protocol

---
