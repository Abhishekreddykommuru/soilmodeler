# 📋 SETUP INSTRUCTIONS

## Spectral Soil Modeler - Detailed Installation Guide

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Running the Application](#running-the-application)
4. [Verifying Installation](#verifying-installation)
5. [Troubleshooting](#troubleshooting)
6. [Additional Configuration](#additional-configuration)

---

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 20.04+)
- **Python**: 3.9 or higher (Python 3.10 or 3.11 recommended)
- **RAM**: 4 GB minimum (8 GB recommended for large datasets)
- **Storage**: 500 MB free space
- **Browser**: Chrome, Firefox, Edge, or Safari (latest versions)

### Checking Python Version
```bash
python --version
# or
python3 --version
```

---

## Installation Steps

### Step 1: Extract the Project

If you received a ZIP file:
```bash
unzip spectral_soil_modeler.zip
cd spectral_soil_modeler
```

Or if cloning from a repository:
```bash
git clone <repository-url>
cd spectral_soil_modeler
```

### Step 2: Create Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Streamlit (web framework)
- Pandas, NumPy (data processing)
- Scikit-learn, SciPy (ML and statistics)
- Plotly, Matplotlib, Seaborn (visualization)
- Python-docx, ReportLab, XlsxWriter (export)
- And other required packages

### Step 5: Verify Installation

```bash
python -c "import streamlit; import sklearn; import plotly; print('All packages installed successfully!')"
```

---

## Running the Application

### Start the Application

```bash
streamlit run app.py
```

### What to Expect

1. Terminal will show:
   ```
   You can now view your Streamlit app in your browser.

   Local URL: http://localhost:8501
   Network URL: http://192.168.x.x:8501
   ```

2. Browser should automatically open to `http://localhost:8501`

3. If browser doesn't open automatically, manually navigate to:
   ```
   http://localhost:8501
   ```

### Stopping the Application

Press `Ctrl+C` in the terminal where the app is running.

---

## Verifying Installation

### Quick Test Procedure

1. **Start the app** - You should see the Dashboard page
2. **Check navigation** - Click through all 6 menu items
3. **Test upload** - Navigate to "Upload & Train" and verify the upload interface appears
4. **Check styling** - Dark theme with gradient sidebar should be visible

### Testing with Sample Data

Create a simple test CSV:

```csv
wavelength_1,wavelength_2,wavelength_3,target
0.5,0.6,0.7,1.2
0.4,0.5,0.6,1.0
0.6,0.7,0.8,1.4
0.3,0.4,0.5,0.8
```

Save as `test_data.csv` and upload to the application.

---

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

### Issue: "streamlit: command not found"

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Reinstall streamlit
pip install streamlit
```

### Issue: Port 8501 already in use

**Solution:**
```bash
# Run on a different port
streamlit run app.py --server.port 8502
```

### Issue: Slow loading or memory errors

**Solutions:**
1. Close other applications
2. Use smaller datasets
3. Reduce number of model combinations

### Issue: Plots not displaying

**Solutions:**
1. Refresh the browser page
2. Try a different browser
3. Clear browser cache

### Issue: Excel files not loading

**Solution:**
```bash
pip install openpyxl xlrd
```

### Issue: Permission denied errors

**Solution (Linux/macOS):**
```bash
chmod +x app.py
```

---

## Additional Configuration

### Changing Theme Colors

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#4A90E2"      # Primary buttons, links
backgroundColor = "#0E1117"    # Main background
secondaryBackgroundColor = "#262B33"  # Cards, sidebar
textColor = "#FAFAFA"         # Main text
```

### Increasing Upload Size Limit

Edit `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 1000  # Size in MB
```

### Running on Network

To allow access from other devices:
```bash
streamlit run app.py --server.address 0.0.0.0
```

### Enabling Wide Mode by Default

Already configured in `app.py`, but can be changed:
```python
st.set_page_config(
    layout="wide",  # or "centered"
)
```

---

## Quick Reference Commands

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py

# Run on custom port
streamlit run app.py --server.port 8502

# Run with specific browser
streamlit run app.py --server.headless true

# Check installed packages
pip list

# Update all packages
pip install --upgrade -r requirements.txt

# Deactivate virtual environment
deactivate
```

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the README.md file
3. Contact the  team at 

---

**Happy Modeling! 🌱**
