# Multitask Action Recognition & Captioning

### **Live Demo:** The model is also live on Hugging Face Spaces: https://huggingface.co/spaces/Tehmas1012/DL_a3

## Project Overview
This project implements a multitask deep learning model for action recognition and image captioning. It consists of a Flask backend serving the PyTorch model and a React frontend (Vite + Tailwind CSS) for the user interface.

## Prerequisites
- Python 3.8+
- Node.js and npm

## Setup and Running

### Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the Flask server:
   ```bash
   python app.py
   ```
   The backend will run on http://localhost:5000.

### Frontend

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
   The frontend will typically run on http://localhost:5173 (check terminal output for the exact URL).

## Usage
1. Open the frontend URL in your browser.
2. Upload an image using the interface.
3. Click "Submit" to get the predicted action and a generated caption.
