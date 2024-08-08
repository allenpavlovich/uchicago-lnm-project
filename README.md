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

- Python 3.10.13
- Anaconda (recommended for environment management)

#### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/allenpavlovich/uchicago-lnm-project.git
   cd uchicago-lnm-project
   ```

2. **Create and activate the conda environment V1:**
   ```sh
   conda env create -f environment.yaml  # Using .yaml extension
   conda env update --file environment_macos.yaml --prune ## ONLY FOR MACOS USERS
   conda activate uchicago_lnm_project
   ```

3. **Creating a Conda Environment Using Anaconda Navigator V2**

To create a conda environment using the provided `environment.yaml` file via Anaconda Navigator, follow these steps:

1. **Open Anaconda Navigator**:
   - Launch Anaconda Navigator from your applications menu or by running `anaconda-navigator` in your terminal.

2. **Go to the Environments Tab**:
   - Click on the "Environments" tab on the left sidebar.

3. **Import the Environment**:
   - Click on the "Import" button located at the bottom of the environments list.

4. **Select the `environment.yaml` File**:
   - In the dialog that appears, click the folder icon to navigate to the location of your `environment.yaml` file.
   - Select the `environment.yaml` file and click "Open".

5. **Name the Environment**:
   - In the "Name" field, enter a name for the new environment (e.g., `YOUR_ENV_NAME`).

6. **Create the Environment**:
   - Click the "Import" button to create the environment from the YAML file.

7. **Activate the Environment**:
   - Once the environment is created, you can activate it via the terminal.
   - Open a terminal window and activate the environment with the following command:
   
     ```sh
     conda activate YOUR_ENV_NAME
     ```

8. **Start Working on Your Project**:
   - Navigate to your project directory if not already there:

     ```sh
     cd /path/to/your/project
     ```

   - Now you can start working on your project with the newly created environment.   

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