# 🌱 Spectral Soil Modeler

**Professional ML Platform for Soil Spectroscopy**



---

## 📋 Overview

Spectral Soil Modeler is an automated machine learning platform designed for predicting soil properties from spectral data. The application provides a user-friendly interface for:

- **Data Upload & Validation**: Smart loading with automatic format detection
- **Automated ML Training**: Train 15 model combinations (3 preprocessing × 5 algorithms)
- **Model Comparison**: Interactive leaderboard and performance visualizations
- **Diagnostics**: Comprehensive residual analysis and PCA
- **Export & Reporting**: Multiple export formats and PDF report generation

---

## ✨ Features

### Preprocessing Methods
- **Reflectance**: Raw spectral data (baseline)
- **Absorbance**: log₁₀(1/R) transformation following Beer-Lambert law
- **Continuum Removal**: Convex hull normalization for absorption features

### ML Algorithms
- **PLSR**: Partial Least Squares Regression
- **Cubist**: Rule-based regression (approximated with Gradient Boosting)
- **GBRT**: Gradient Boosted Regression Trees
- **KRR**: Kernel Ridge Regression
- **SVR**: Support Vector Regression

### Performance Metrics
- R² (Coefficient of Determination)
- RMSE (Root Mean Square Error)
- RPD (Ratio of Performance to Deviation)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Correlation (Pearson)
- Bias

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone or extract the project:**
   ```bash
   cd spectral_soil_modeler
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser:**
   The app will automatically open at `http://localhost:8501`

---

## 📖 User Guide

### 1. Upload Data

1. Navigate to **Upload & Train** page
2. Upload your spectral data file (CSV, XLS, or XLSX)
3. The system automatically:
   - Detects file format
   - Handles encoding issues
   - Validates data quality
4. Select your target property column
5. Review data statistics and quality indicators

### 2. Configure Training

1. Go to the **Configuration** tab
2. Select preprocessing methods (default: all 3)
3. Select ML algorithms (default: all 5)
4. Expert mode allows:
   - Custom train/test split
   - Cross-validation folds
   - Random seed setting

### 3. Train Models

1. Navigate to the **Training** tab
2. Click "Start Automated Training"
3. Monitor progress in real-time
4. View instant results upon completion

### 4. Analyze Results

- **Model Comparison**: View leaderboard, visualizations, residual analysis
- **Diagnostics**: Feature importance, PCA analysis, detailed diagnostics
- **Model Archive**: Access training history and stored models

### 5. Export Results

- Export to Excel, CSV, or JSON
- Download trained model files
- Generate PDF reports

---

## 📁 Project Structure

```
spectral_soil_modeler/
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── SETUP_INSTRUCTIONS.md       # Detailed setup guide
│
├── .streamlit/
│   └── config.toml            # Streamlit configuration
│
├── components/                 # UI page components
│   ├── __init__.py
│   ├── dashboard.py           # Home/overview page
│   ├── upload_train.py        # Data upload and training
│   ├── model_comparison.py    # Results comparison
│   ├── model_archive.py       # Training history
│   ├── diagnostics.py         # Advanced diagnostics
│   └── analytics_export.py    # Export and reporting
│
├── modules/                    # Core logic modules
│   ├── __init__.py
│   ├── data_loader.py         # File loading and validation
│   ├── preprocessing.py       # Spectral preprocessing
│   ├── models.py              # ML model definitions
│   ├── training.py            # Training orchestration
│   ├── evaluation.py          # Performance evaluation
│   ├── visualization.py       # Plotly visualizations
│   └── utils.py               # Utility functions
│
├── styles/
│   └── custom.css             # Custom CSS styles
│
├── models/                     # Saved trained models
├── reports/                    # Generated reports
├── cache/                      # Temporary cache
└── logs/                       # Application logs
```

---

## 📊 Data Format

### Supported Formats
- CSV (comma, semicolon, tab, or pipe separated)
- Excel (XLS, XLSX)

### Expected Structure
- **Rows**: Samples
- **Columns**: Spectral bands + Target property
- Spectral band columns can be:
  - Numeric wavelengths (e.g., 350, 351, ...)
  - Wavelength with units (e.g., 350nm, 351nm, ...)
  - Any column names for features

### Example Data Structure

| Sample_ID | 350 | 351 | ... | 2500 | SOC |
|-----------|-----|-----|-----|------|-----|
| S001 | 0.45 | 0.46 | ... | 0.32 | 2.3 |
| S002 | 0.42 | 0.43 | ... | 0.29 | 1.8 |
| ... | ... | ... | ... | ... | ... |

---

## ⚙️ Configuration

### Application Settings

Edit `.streamlit/config.toml` to customize:

```toml
[theme]
primaryColor = "#4A90E2"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262B33"
textColor = "#FAFAFA"

[server]
maxUploadSize = 500
```

### Training Defaults

Modify in `modules/training.py`:
- Test split ratio: 20%
- Cross-validation folds: 5
- Random seed: 42

---

## 🔧 Troubleshooting

### Common Issues

1. **Import errors**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Memory issues with large files**
   - Reduce number of samples
   - Use chunked processing

3. **Slow training**
   - Reduce number of model combinations
   - Lower cross-validation folds

4. **Visualization not displaying**
   - Refresh the browser
   - Check browser compatibility

---

## 📝 Requirements

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
plotly>=5.17.0
matplotlib>=3.7.0
seaborn>=0.12.0
python-docx>=0.8.11
reportlab>=4.0.0
xlsxwriter>=3.1.0
openpyxl>=3.1.0
xlrd>=2.0.1
joblib>=1.3.0
pyyaml>=6.0
```

---

## 👥 Credits

**Developed by:**
- Software Engineering Research Center ()
- For Language Speech Interface Lab (LSI)
- 

**Version:** 1.0.0

---

## 📄 License

This project is developed for academic and research purposes at .

---

## 📧 Contact

For support or questions, please contact the  team at .
