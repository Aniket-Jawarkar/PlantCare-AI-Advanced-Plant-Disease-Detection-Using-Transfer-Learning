# Prerequisites and Setup Guide

This document outlines the necessary prerequisites and environment setup required to run the PlantCare-AI ML Pipeline and Web Application.

## System Requirements
- **Operating System**: Windows (tested), Linux, or macOS
- **Python Version**: Python 3.8 to 3.10 (TensorFlow compatibility)
- **Disk Space**: At least 2GB of free space for datasets and models.

## Environment Setup

It is highly recommended to use a virtual environment to manage dependencies.

### 1. Create a Virtual Environment
```bash
# Using standard venv
python -m venv venv

# Activate the environment (Windows)
.\venv\Scripts\activate

# Activate the environment (Linux/macOS)
source venv/bin/activate
```

### 2. Install Dependencies
Install the required Python packages. You can install them manually or use a `requirements.txt` if available.

```bash
pip install tensorflow keras numpy pandas matplotlib pillow flask werkzeug scikit-learn
```

**Key Libraries**:
- `tensorflow` / `keras`: For loading and running the MobileNetV2 transfer learning model.
- `flask`: For the web application backend.
- `pillow`: For image processing.
- `numpy` / `pandas`: For data manipulation.
- `scikit-learn`: For calculating model metrics (classification report).

## Dataset Requirements
Before running the modeling pipeline, you must have the dataset properly placed in the `1_Data_Collection_and_Preprocessing` folder.

The data should be organized into `train` and `valid` (and optionally `test`) directories:

```text
1_Data_Collection_and_Preprocessing/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   └── ... (38 classes total)
└── valid/
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    └── ... (38 classes total)
```

Ensure your data is in place before proceeding to the [Workflow](../0_Workflow/README.md).
