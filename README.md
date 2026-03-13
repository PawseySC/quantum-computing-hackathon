# Quantum Computing Hackathon 

This repo provides material to introduce Quantum Computing to students in secondary school. The lessons are laid out in the `lessons/` directory in the form of interactive, digital (jupyter) notebooks.  

For a tutorial on what a Jupyter notebook looks like and how to use it, have a look at (this)[https://realpython.com/jupyter-notebook-introduction/].

## Setting up the notebooks

The digital notebooks make use of quite a variety of software behind the scenes. We have setup executables (scripts) to run that will install all the relevant software packages save Python. To install Python, we suggest looking at this (material)[https://realpython.com/installing-python/].

There are several ways of downloading this hackathon. 

### Simple

The first step is download the repository and unpack it. The repository can be downloaded from (here)[https://github.com/PawseySC/quantum-computing-hackathon/archive/refs/heads/main.zip]. Once you unzip it, you should have a folder/directory called `quantum-computing-hackathon`. In this folder you see a scripts directory. 

#### For Windows

Run the executable `scripts/Launch_Windows.bat`. 

This will begin installing the needed software before opening up the Jupyter interface in a browser.

#### For Linux/MacOS

Open a terminal and go to this directory. Once there simply run
```bash
./scripts/Launch_Linux.sh
```
or MacOS as appropriate. 

This will begin installing the needed software before opening up the Jupyter interface in a browser.

### Advanced Users

For more advanced users using Linux or Mac, we suggest using git clone from a terminal and running the relevant scripts. 

```bash
git clone https://github.com/PawseySC/quantum-computing-hackathon.git
cd quantum-computing-hackathon
./scripts/Launch_Linux.sh
```

This will create a Python virtual environment, use `pip` to install relevant packages and then launch Jupyter using the default browser. 

## Lessons

The notebooks are in `lessons/` and consist of 

* `0-Primer_to_Quantum_Computing`: a non-interactive lesson that introduces 
  - programming concepts with a focus on the Python programming language
  - classical and quantum bits
  - quantum processes
  - quantum circuits (a way of visualising quantum computing algorithms)
* `1-Intro_to_Quantum_Computing`: a interactive lesson that covers
  - quantum computing basics
  - simple quantum operations 
* `2-Intro_Quantum_Algorithms`: a interactive lesson that covers
  - quantum computing algorithms 
