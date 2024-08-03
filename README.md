# UChicago LNM Project

This repository contains the project files for the Linear and Nonlinear Models course at the University of Chicago.

## Repository Structure

```plaintext
├── data/
│   ├── raw/                # Raw data files
│   ├── processed/          # Processed data files ready for analysis
├── notebooks/
│   ├── causal_inference/   # Notebooks for causal inference analyses
│   ├── exploratory/        # Notebooks for exploratory data analysis (EDA)
│   ├── modeling/           # Notebooks for model development and tuning
│   ├── xai/                # Notebooks for Explainable AI (XAI) techniques
├── reports/
│   ├── figures/            # Figures and plots for reports
│   ├── final_report/       # Final project report
│   ├── presentations/      # Slide decks for presentations
├── src/
│   ├── causal_inference/   # Scripts for causal inference methods
│   ├── data/               # Scripts for data preprocessing
│   ├── features/           # Scripts for feature engineering
│   ├── models/             # Scripts for training and evaluating models
│   ├── xai/                # Scripts for XAI methods (LIME, SHAP & etc.)
├── tests/                  # Unit tests and integration tests
├── environement.yaml       # Environment and list of dependencies
├── README.md               # Project overview and setup instructions
├── .gitignore              # Git ignore file
├── .gitattributes          # Git attributes file

```

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
   conda activate uchicago_lnm_project

## Running Tests

To run the unit tests for this project, follow these steps:

1. Ensure you have the Conda environment activated:
    ```sh
    conda activate uchicago_lnm_project
    ```

2. Navigate to the root directory of the project.

3. Run the tests using the following command:
    ```sh
    python -m unittest discover -s tests -v
    ```

This will discover and run all the test cases in the `tests` directory.

## Contribution Guidelines

### Branch Naming Conventions

1. **Feature Branches**:
   - Naming convention: `feature/description`
   - Example: `feature/add-xai-methods`
   
2. **Bugfix Branches**:
   - Naming convention: `bugfix/description`
   - Example: `bugfix/fix-data-loading-error`

3. **Experimental Branches**:
   - Naming convention: `experiment/description`
   - Example: `experiment/test-new-model`

### Workflow

1. **Create a Branch**:
   ```sh
   git checkout -b feature/new-analysis
   ```

2. **Make Changes and Commit**:
   ```sh
   git add .
   git commit -m "Add exploratory data analysis for sentiment distribution"
   ```

3. **Push Branch**:
   ```sh
   git push origin feature/new-analysis
   ```

4. **Create a Pull Request**:
   - Go to the repository on GitHub.
   - Click on "Compare & pull request".
   - Add a description of the changes and assign reviewers.
   - Submit the pull request for review.

5. **Review and Merge**:
   - Team members review the pull request.
   - Address any feedback and make necessary changes.
   - Once approved, merge the pull request into the `main` branch.

### Example Commands
   ```sh
    # Create a new branch for your feature
    git checkout -b feature/add-xai-methods

    # Make changes and commit
    git add .
    git commit -m "Implement SHAP and LIME for model explainability"

    # Push the branch to the remote repository
    git push origin feature/add-xai-methods

    # Create a pull request on GitHub and request a review
    # Once approved, merge the pull request
   ```