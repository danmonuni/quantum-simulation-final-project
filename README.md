# AI Models for Physics Final Project

Final project for the course on Quantum Simulation.

The project starts from the paper: [Learning ground states of quantum Hamiltonians with graph networks](https://arxiv.org/abs/2110.06390).
It tries to replicate the paper's results in simplified and increasingly more complex settings.

## 🚀 Run it in Colab

Click the badge below to open and run the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danmonuni/quantum-simulation-final-project/blob/main/quantum_simulation_notebook.ipynb)

## 📊 Results

The full study with results is documented both in a slide deck accessible below:

[![View Slides](https://img.shields.io/badge/Slides-Presentation-orange)](https://colab.research.google.com/github/danmonuni/quantum-simulation-final-project/blob/main/quantum_slides.pdf)





how-to-use: 
    other folders contain the modular components of the project shall I develop it further
    in the notebook section there is a complete and functioning notebook

approach: for learning reasons, I will follow an incremental approach with a diamond shape.
    root: study a probabilistic mixture on a small lattice so that i can use an exhaustive energy evaluation
    branch_quantum: study a quantum superposition on a small lattice (exhaustive evaluation)
    branch_mcmc: study a probabilistic mixture with mcmc evalutaion
    merge: study a quantum superposition with mcmc evaluation
