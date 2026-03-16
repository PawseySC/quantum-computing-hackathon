#!/bin/bash

# set the pipefail option to ensure that if any command in a pipeline fails, the entire pipeline will be considered to have failed. 
set -euo pipefail

# determine the directory where scripts are located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Report what script will do 
echo "This script will "
echo "\\t * create a Python virtual environment"
echo "\tt * install the required packages from requirements.txt"
echo "\tt * launch Jupyter Notebook on port 8888."

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
VENV_DIR=${SCRIPT_DIR}"/../venv"

# Check if venv directory already exists
if [ -d "$VENV_DIR" ]; then
    read -p "Virtual environment 'venv' already exists. Do you want to delete it and recreate it? (y/n) " choice
    choice=$(echo $choice | tr 'a-z' 'A-Z')
    
    if [ "$choice" = "Y" ] || [ "$choice" = "YES" ]; then
        echo "Deleting existing virtual environment..."
        rm -rf "$VENV_DIR"
        echo "Creating new virtual environment..."
        $python_exec -m venv "$VENV_DIR" || { echo "Failed to create virtual environment"; exit 1; }
    else
        echo "Keeping existing virtual environment."
    fi
fi


# Activate the virtual environment
source "$VENV_DIR"/bin/activate > /dev/null 2>&1

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
    rm -rf "$VENV_DIR" || true
    exit 1
fi

echo "Packages installed successfully."

# --- 4. Launch Jupyter Notebook ---
echo "Starting Jupyter Notebook..."

# Add this line to ensure Jupyter works with modern browsers
# jupyter nbconvert --to widget_link --output jupyter_notebook_config.json --config

jupyter notebook lessons/ --port=8888 &

# Optional: Wait for Jupyter to finish (useful if you want to run commands after)
wait

# deactive the python environment after Jupyter is closed
deactivate > /dev/null 2>&1
