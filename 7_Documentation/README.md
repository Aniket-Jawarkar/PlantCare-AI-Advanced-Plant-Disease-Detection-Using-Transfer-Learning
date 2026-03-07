# Documentation

Welcome to the Documentation directory for **PlantCare-AI: Advanced Plant Disease Detection Using Transfer Learning**. 

This folder is designated for storing all comprehensive documentation related to the project. It serves as the single source of truth for the system's architecture, ML pipelines, API references, and deployment guides.

## Table of Contents
1. [Project Architecture](#project-architecture)
2. [Machine Learning Pipeline](#machine-learning-pipeline)
3. [Web Application & API](#web-application--api)
4. [File Structure](#file-structure)

---

## Project Architecture
PlantCare-AI leverages deep learning, specifically Transfer Learning techniques, to accurately classify plant diseases from images of plant leaves. The system consists of:
- **Data Module**: Handles ingestion, cleaning, and augmentation of raw plant leaf datasets.
- **Modeling Module**: Defines the neural network architecture (e.g., utilizing pre-trained models like ResNet, VGG, or MobileNet).
- **Inference/Web Module**: A web-based user interface (accessible via `localhost:5000`) where users can upload leaf images to receive instant disease predictions and care recommendations.

## Machine Learning Pipeline
The pipeline is fully automated and can be executed via the `run_pipeline.ps1` script found in the project root. The phases included are:
1. **Data Preprocessing** (`1_Data_Collection_and_Preprocessing`): Normalizes images, splits data into training/validation sets, and applies data augmentation.
2. **Model Building** (`2_Model_Building`): Constructs the transfer learning model by freezing base layers and appending custom classification heads suited for the dataset's classes.
3. **Model Training** (`3_Model_Training`): Fits the model to the training dataset, utilizes callbacks like Early Stopping or Model Checkpointing.
4. **Evaluation & Testing** (`4_Model_Evaluation_and_Testing`): Measures model performance (Accuracy, Precision, Recall, F1-Score) on unseen data and generates confusion matrices.

## Web Application & API
The frontend and backend of the application reside in `5_Application_Building`. 
- **Tech Stack**: Built with a backend framework exposing APIs to a web frontend.
- **Functionality**: Provides an endpoint for image upload which is then passed to the trained saved model (stored in `6_Models_and_Outputs`).
- **Running Locally**: The web server is launched in the final step of `run_pipeline.ps1` and runs on `http://localhost:5000`.

## File Structure
Any additional specific technical documents should be placed within this `7_Documentation` folder. Suggested future documents to include:
- `API_Reference.md`: Detailed breakdown of the web application endpoints.
- `Data_Dictionary.md`: Definitions of categories/classes used in the dataset.
- `Model_Cards.md`: Information describing model performance metrics, biases, and limitations.

*(Note: Keep this directory updated as the project evolves!)*
