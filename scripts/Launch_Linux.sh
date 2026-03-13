#!/bin/bash
set -euo pipefail

# --- 1. Determine where Python 3 is located ---
python_exec=$(command -v python3)

if [ $? -ne 0 ]; then
    echo "ERROR: Could not locate a Python 3 interpreter."
    echo "Please ensure Python 3 is installed and available in your PATH."
    exit 1
fi

echo "Found Python at: $python_exec"

# --- 2. Derive the pip path from the python executable ---
# This assumes a standard Python installation
pip_exec="$python_exec -m pip"

echo "Found pip at: $pip_exec"

# --- 3. Create Virtual Environment and Install Packages ---
venv_dir="./venv"

# Check if venv directory already exists
if [ -d "$venv_dir" ]; then
    read -p "Virtual environment 'venv' already exists. Do you want to delete it and recreate it? (y/n) " choice
    choice=$(echo $choice | tr 'a-z' 'A-Z')
    
    if [ "$choice" = "Y" ] || [ "$choice" = "YES" ]; then
        echo "Deleting existing virtual environment..."
        rm -rf "$venv_dir"
        echo "Creating new virtual environment..."
        $python_exec -m venv "$venv_dir" || { echo "Failed to create virtual environment"; exit 1; }
    else
        echo "Keeping existing virtual environment."
    fi
fi


# Activate the virtual environment
source "$venv_dir"/bin/activate > /dev/null 2>&1

# Now, pip is available in the activated environment
echo "Creating virtual environment and installing packages..."
$python_exec -m pip install --ignore-installed --no-cache-dir -r "requirements.txt" || { echo "Failed to install packages from requirements.txt"; exit 1; }

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install packages from requirements.txt."
    echo "Please check that a requirements.txt file exists and is correctly formatted."
    echo "It is also possible that there is an issue with:"
    echo "\\t - Network connectivity"
    echo "\\t - the Python installation"
    echo "\\t - Compilers or build tools required for some packages"
    # Deactivate and clean up
    deactivate > /dev/null 2>&1
    rm -rf "$venv_dir" || true
    exit 1
fi

echo "Packages installed successfully."

# --- 4. Launch Jupyter Notebook ---
echo "Starting Jupyter Notebook..."

# Add this line to ensure Jupyter works with modern browsers
jupyter nbconvert --to widget_link --output jupyter_notebook_config.json --config

jupyter notebook --port=8888 &

# Optional: Wait for Jupyter to finish (useful if you want to run commands after)
# Uncomment the next line if you need this
# wait

deactivate > /dev/null 2>&1
