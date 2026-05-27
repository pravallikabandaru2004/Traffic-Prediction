# Traffic Prediction and Congestion Analysis Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey?logo=flask)
![ML](https://img.shields.io/badge/Machine%20Learning-Enabled-green)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚦 Project Overview

This project is a professional-grade web dashboard for real-time traffic prediction and congestion analysis. It leverages advanced machine learning models to forecast traffic flow, visualize congestion hotspots, and provide actionable insights for urban mobility management.

---

## ✨ Features

- **Traffic Flow Prediction:** Accurate short-term and long-term traffic forecasting using deep learning models.
- **Interactive Dashboard:** Visualizes live and historical traffic data, congestion levels, and sensor analytics.
- **Congestion Analysis:** Identifies and highlights congestion hotspots with severity metrics.
- **User Authentication:** Secure login and registration for personalized dashboard access.
- **Data Upload:** Supports uploading new sensor and traffic datasets for custom analysis.
- **Model Performance Metrics:** Displays accuracy, loss, and other evaluation metrics.

---

## 🔄 Traffic Prediction Workflow

1. **Data Ingestion:** Reads traffic, sensor, and adjacency data from CSV files.
2. **Preprocessing:** Cleans and transforms time series data for model input.
3. **Model Inference:** Loads pre-trained models (e.g., ST-MLP) from `saved_models/` and predicts future traffic sequences.
4. **Visualization:** Renders predictions and congestion maps on the dashboard.
5. **Analysis:** Computes and displays congestion severity and model accuracy metrics.

---

## 🛠️ Technologies Used

| Layer         | Technology                |
|-------------- |--------------------------|
| Backend       | Python, Flask             |
| ML Models     | PyTorch                   |
| Frontend      | HTML5, CSS3, JavaScript   |
| Visualization | Chart.js, Bootstrap, jQuery|
| Data          | CSV (traffic, sensors)    |

---

## 📊 Dataset & Model Details

- **Datasets:**
  - `traffic_sequences.csv`, `traffic_time_series.csv`: Traffic flow data
  - `sensors.csv`: Sensor metadata
  - `adjacency_edges.csv`: Road network graph
- **Models:**
  - Pre-trained models (e.g., ST-MLP) stored in `saved_models/`
  - Model code in `model.py`, training scripts in `training.py`

---

## 📈 Dashboard Functionality

- **Home:** Overview and quick stats
- **Analysis:** Visualizes congestion, sensor data, and predictions
- **Prediction:** Run new predictions and view results
- **Authentication:** Login/Register for personalized features

---

## 🚦 Congestion Analysis & Accuracy Metrics

- **Congestion Severity:**
  - Visual heatmaps and severity scores for each sensor/road segment
- **Model Accuracy:**
  - Metrics such as MAE, RMSE, and accuracy displayed on dashboard

---

## ⚡ Local Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/pravallikabandaru2004/Traffic-Prediction
cd traffic-copy
```

### 2. Python Environment Setup

```bash
# (Recommended) Create a virtual environment
python -m venv venv
# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
# If requirements.txt is missing, install manually:
pip install flask torch pandas numpy scikit-learn matplotlib
```

### 4. (Optional) Install Node.js Dependencies

If you have a frontend build process (e.g., for advanced JS/CSS):

```bash
npm install
```

### 5. Environment Variables

Set any required environment variables (if needed):

```bash
# Example (if needed):
set KMP_DUPLICATE_LIB_OK=TRUE
```

### 6. Run the Backend (Flask Server)

```bash
python app.py
```

The backend will start at `http://127.0.0.1:5000/` by default.

### 7. Run the Frontend

- Static HTML/CSS/JS: Open `static/index.html` or access via Flask routes.
- If using a frontend framework (React/Vue/etc.):
  ```bash
  npm start
  ```

### 8. Access the Dashboard

Open your browser and go to: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 🧪 Testing & Usage

- **Run Predictions:** Use the dashboard to upload data and run predictions.
- **View Analysis:** Explore congestion maps and model metrics.
- **Test Model:** Modify `training.py` to retrain or test models with new data.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Contact

For questions or support, please open an issue or contact the maintainer.
