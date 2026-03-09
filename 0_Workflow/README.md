# Execution Workflow

This guide details the end-to-end execution of the PlantCare-AI project, from data preprocessing to launching the web application.

## Overview

The project is structured into distinct sequential stages:
1. Data Preprocessing
2. Model Building
3. Model Training
4. Model Evaluation
5. Application Launch

You can run the entire pipeline automatically using the provided PowerShell script, or run each step manually.

---

## Option 1: Automated Pipeline (Windows)

A single PowerShell script (`run_pipeline.ps1`) is provided in the root directory to execute all steps sequentially.

```powershell
# In the root directory of the project
.\run_pipeline.ps1
```
*(Note: Ensure your execution policy permits running scripts: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

---

## Option 2: Step-by-Step Manual Execution

If you prefer to run steps individually (e.g., for debugging or partial retraining), follow these steps:

### Step 1: Data Preprocessing
**Directory**: `1_Data_Collection_and_Preprocessing`
**Script**: `preprocessing.py` OR `eda.py`

This step verifies the dataset structure, performs exploratory data analysis (EDA), and sets up the image data generators with augmentation.
```bash
cd 1_Data_Collection_and_Preprocessing
python preprocessing.py
# optional: python eda.py to visualize data
cd ..
```

### Step 2: Model Building
**Directory**: `2_Model_Building`
**Script**: `build_model.py`

This step constructs the neural network architecture using MobileNetV2 as the base model for transfer learning. It sets up the custom classification head for 38 classes.
```bash
cd 2_Model_Building
python build_model.py
cd ..
```

### Step 3: Model Training
**Directory**: `3_Model_Training`
**Script**: `train_model.py`

This script loads the preprocessed data and the compiled model architecture, then trains the model. It saves the best weights (`plant_disease_best.h5` or `plant_disease_saved_model`) to the `6_Models_and_Outputs` folder.
```bash
cd 3_Model_Training
python train_model.py
cd ..
```

### Step 4: Model Evaluation
**Directory**: `4_Model_Evaluation_and_Testing`
**Script**: `evaluate_model.py`

This step evaluates the trained model against the validation dataset. It generates a comprehensive `classification_report.json` which is saved in `6_Models_and_Outputs` and used by the web application for class labels.
```bash
cd 4_Model_Evaluation_and_Testing
python evaluate_model.py
cd ..
```

### Step 5: Run the Web Application
**Directory**: `5_Application_Building`
**Script**: `app.py`

This launches the Flask user interface, allowing users to upload plant leaf images and receive immediate disease predictions.
```bash
cd 5_Application_Building
python app.py
```
*Access the application at `http://localhost:5000` in your web browser.*
