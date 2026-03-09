# 🌿 PlantCare-AI 

**Advanced Plant Disease Detection Using Transfer Learning**

PlantCare-AI is an intelligent, end-to-end machine learning system designed to automatically identify plant diseases from leaf images. By leveraging **MobileNetV2** through transfer learning, the system accurately classifies images across **38 unique crop and disease categories**. It features a modern, user-friendly Flask REST web application for instant inference.

![Contributions welcome](https://img.shields.io/badge/Contributions-welcome-brightgreen.svg)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

---

## 📖 Table of Contents
1. [Project Structure](#-project-structure)
2. [Quick Start](#-quick-start)
3. [Documentation Guides](#-documentation-guides)
4. [Classes Supported](#-classes-supported)

---

## 📂 Project Structure

The repository is modularly designed, following standard ML pipeline stages:

- **`0_Prerequisites/`**: Environment setup and dependency requirements.
- **`0_Workflow/`**: Step-by-step pipeline execution guide.
- **`1_Data_Collection_and_Preprocessing/`**: Scripts for data augmentation and exploratory data analysis.
- **`2_Model_Building/`**: Architecture definition (MobileNetV2 transfer learning).
- **`3_Model_Training/`**: Training scripts and checkpoint configuration.
- **`4_Model_Evaluation_and_Testing/`**: Model evaluation against validation sets and report generation.
- **`5_Application_Building/`**: Flask web application for uploading images and viewing predictions.
- **`6_Models_and_Outputs/`**: Directory where trained models (`.h5` / SavedModel) and metric reports are stored.
- **`7_Documentation/`**: Auxiliary project documentation.
- **`8_Deployment/`**: Guides for deploying the web application to production.
- **`9_Future_Implementations/`**: Plans and placeholders for future extensions.
- **`run_pipeline.ps1`**: Automated PowerShell script to run the entire project end-to-end.

---

## 🚀 Quick Start

To instantly launch the pipeline (assuming [prerequisites](0_Prerequisites/README.md) are met and datasets are present):

**Windows (PowerShell)**:
```powershell
.\run_pipeline.ps1
```
This script sequentially preprocesses data, builds the model, trains it, evaluates accuracy, and automatically launches the Flask web app at `http://localhost:5000`.

---

## 📚 Documentation Guides

For detailed, step-by-step instructions, please consult the dedicated READMEs inside the respective directories:

- **[Environment Setup & Prerequisites](0_Prerequisites/README.md)**: What you need before running the project.
- **[Detailed Execution Workflow](0_Workflow/README.md)**: How the ML pipeline executes logically.
- **[Deployment Instructions](8_Deployment/README.md)**: How to host the flask application locally or in production.

---

## 🗂 Classes Supported

The model is trained on the standard PlantVillage dataset, capable of recognizing 38 distinct plant and disease condition combinations, including:
- **Apple**: Scab, Black Rot, Cedar Apple Rust, Healthy
- **Corn (Maize)**: Cercospora, Common Rust, Northern Leaf Blight, Healthy
- **Potato**: Early Blight, Late Blight, Healthy
- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy
- *And many more...*