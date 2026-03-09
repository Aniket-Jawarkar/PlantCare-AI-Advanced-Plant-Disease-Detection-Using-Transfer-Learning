# Deployment Guide

This document explains how to deploy the PlantCare-AI web application. 

The application is built using **Flask** and **TensorFlow**, making it suitable for both local usage and production deployment.

## Local Deployment (Development)

To run the application locally for development or testing purposes:

1. Ensure the model (`plant_disease_saved_model` or `plant_disease_best.h5`) and the `classification_report.json` exist in the `6_Models_and_Outputs` directory.
2. Navigate to the application folder:
   ```bash
   cd 5_Application_Building
   ```
3. Run the Flask development server:
   ```bash
   python app.py
   ```
4. Open your browser and go to `http://localhost:5000` or `http://127.0.0.1:5000`.

> [!WARNING]
> The default Flask server is designed only for development and debugging. It is not secure or efficient enough for production.

---

## Production Deployment

For deploying to a production environment (e.g., a VPS, AWS, Heroku, Render), you should use a production-grade WSGI server like **Gunicorn** (Linux/macOS) or **Waitress** (Windows).

### Linux / macOS (Using Gunicorn)

1. **Install Gunicorn**:
   ```bash
   pip install gunicorn
   ```
2. **Run the Application**:
   Navigate to `5_Application_Building` and start the server:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8000 app:app
   ```
   * `-w 4` sets the number of worker processes (adjust based on your server's CPU cores).
   * `-b 0.0.0.0:8000` binds the server to port 8000.

### Windows (Using Waitress)

Gunicorn is not supported on Windows. Use Waitress instead.

1. **Install Waitress**:
   ```bash
   pip install waitress
   ```
2. **Create a script (e.g., `run_prod.py`)**:
   ```python
   from waitress import serve
   from app import app, load_model

   if __name__ == '__main__':
       load_model()
       serve(app, host='0.0.0.0', port=8000)
   ```
3. **Run the server**:
   ```bash
   python run_prod.py
   ```

### Important Considerations for Deployment
- **Memory Limit**: TensorFlow models can consume significant RAM. Ensure your hosting environment has at least 1GB - 2GB of RAM available.
- **File Uploads**: The application receives image uploads. Ensure the deployment environment allows write access to the `uploads/` and `static/images/` directories, and consider implementing scheduled cleanups for stored images.
- **HTTPS**: For secure data transmission, place your WSGI server behind a reverse proxy like Nginx or configure SSL/TLS directly if supported by your hosting provider.
