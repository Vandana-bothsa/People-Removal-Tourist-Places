# People Removal from Tourist Places

This project focuses on automatically removing people from tourist place images
to generate clean and crowd-free photographs using deep learning techniques.

---

## Project Overview
Tourist places are often crowded, which affects the visual quality of images.
This project detects people in tourist images and removes them while preserving
the background, producing clean images suitable for tourism promotion and
photography enhancement.

---

## Objectives
- Detect people in tourist place images
- Remove detected people automatically
- Reconstruct the background using image inpainting
- Generate clean and visually appealing images

---

## Workflow
1. Input tourist images are provided to the system  
2. People are detected using YOLOv8 segmentation  
3. Segmentation masks are generated for detected people  
4. Inpainting is applied to remove people and fill the background  
5. Clean output images are saved  

---

## Technologies Used
- Python
- YOLOv8 (Segmentation)
- OpenCV
- NumPy
- Google Colab

---

## Dataset
The dataset consists of tourist place images collected from publicly available
online sources.  
Due to size and licensing constraints, the complete dataset is not included in
this repository. Sample input and output images are provided for demonstration.

---

## Results
The system successfully removes people from multiple tourist images and produces
clean background images with minimal visual artifacts.

---

## Applications
- Tourism and travel photography
- Image enhancement for social media
- Smart tourism applications
- Computer vision research

---

## Conclusion
This project demonstrates the effective use of deep learning-based segmentation
and image inpainting for real-world image editing tasks. The approach can be
extended to video processing and real-time applications.

---

## Author
Vandana Bothsa  
B.Tech – Computer Science and Engineering


## Folder Structure
