# Automated Neonatal Hip Ultrasound System for Diagnosing Developmental Dysplasia of the Hip Using Assistive AI

## Overview
This repository contains the implementation of an **AI-based diagnostic system** for **developmental dysplasia of the hip (DDH)** in **infant hip ultrasonography**. The system leverages **deep learning models** for **standard plane classification, Graf region segmentation, and DDH grading**, aiming to provide an **automated, efficient, and accurate diagnostic solution** integrated into portable ultrasound devices.

## Key Features
- **Standard Plane Classification:** Differentiates between standard and non-standard hip ultrasound planes.
- **Graf Region Segmentation:** Accurately identifies key anatomical regions for DDH evaluation.
- **DDH Grading System:** Classifies hip abnormalities based on the Graf method.
- **Optimized for Mobile Devices:** Implemented using **TensorFlow Lite** for real-time operation on smartphones.
- **AI Models Used:** NASNetMobile, MobileNetV2, UNet, NestedUNet, DeepLabV3Plus, and PAN.

## Dataset
The dataset was collected using the **Healcerion Inc. SONON 300L** device, focusing on babies under one year old. It includes **ultrasound images and expert-labeled annotations**.

### Dataset Composition
- **Standard Plane Classification Dataset:** 39,349 images.
- **Graf Region Segmentation Dataset:** 4,411 images.
- **DDH Grading Dataset:** 4,411 segmented images classified into three grades based on Graf angles.

**Ethical Approval:** This study was reviewed and approved by the **Institutional Review Board (IRB) of Gachon University (Approval Number: GCIRB2020-477)**.

## Model Architecture
The AI models consist of:
1. **Standard Plane Classification Model**:
   - NASNetMobile achieved the highest AUC of **0.864**.
   - Other tested models: MobileNetV2, MobileNetV1, DenseNet121, EfficientNetV2B0, and ResNet50.
2. **Graf Region Segmentation Model**:
   - UNet demonstrated the best **Dice coefficient of 0.794**.
   - Other tested models: NestedUNet, DeepLabV3Plus, and PAN.
3. **DDH Grading Model**:
   - NestedUNet achieved the best F1-score of **0.804** for detecting **subluxation and dislocation (Grade 3)**.
![figure4](https://github.com/user-attachments/assets/c3246224-bbc4-47d2-aba3-c982cee11bac)

## Performance Metrics
The AI models were evaluated using **AUC, Dice coefficient, recall, precision, sensitivity, and specificity**. Performance comparisons for segmentation and classification tasks were conducted using different AI architectures.

## Installation
Clone this repository and install dependencies:
```bash
git clone https://github.com/your-repo/ddh-ultrasound.git
cd ddh-ultrasound
pip install -r requirements.txt
```

## Ethical Statement
- **Approval:** This study was reviewed and approved by the Institutional Review Board (IRB) (Approval Number: GCIRB2020-477).
- **Data Privacy:** The dataset is available upon request with appropriate permissions.

## Citation
If you use this code, please cite:
```bibtex
@article{lee2025ddh,
  author = {Your Name et al.},
  title = {Automated Neonatal Hip Ultrasound System for Diagnosing Developmental Dysplasia of the Hip Using Assistive AI},
  journal = {Your Journal},
  year = {2025}
}
```
## License
This project is licensed under the MIT License.
