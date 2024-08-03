# UChicago LNM Project

This repository contains the project files for the Linear and Nonlinear Models course at the University of Chicago.

## Repository Structure

├── data/
│ ├── raw/ # Raw data files
│ ├── processed/ # Processed data files ready for analysis
│ ├── external/ # External datasets or additional data
├── notebooks/
│ ├── exploratory/ # Notebooks for exploratory data analysis (EDA)
│ ├── modeling/ # Notebooks for model development and tuning
│ ├── xai/ # Notebooks for Explainable AI (XAI) techniques
│ ├── causal_inference/ # Notebooks for causal inference analyses
├── src/
│ ├── data/ # Scripts for data preprocessing
│ ├── features/ # Scripts for feature engineering
│ ├── models/ # Scripts for training and evaluating models
│ ├── xai/ # Scripts for XAI methods (LIME, SHAP)
│ ├── causal_inference/ # Scripts for causal inference methods
├── reports/
│ ├── figures/ # Figures and plots for reports
│ ├── presentations/ # Slide decks for presentations
│ ├── final_report/ # Final project report
├── tests/ # Unit tests and integration tests
├── requirements.txt # List of dependencies
├── README.md # Project overview and setup instructions
├── .gitignore # Git ignore file


### Getting Started

#### Prerequisites

- Python 3.8+
- Anaconda (recommended for environment management)

#### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/allenpavlovich/uchicago-lnm-project.git
   cd uchicago-lnm-project

2. **Create and activate the conda environment:**
   ```sh
   conda env create -f environment.yml
   conda activate lnm_project