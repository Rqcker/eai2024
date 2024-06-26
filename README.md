### Topic Modelling research in the Digital Circular Electrochemical Economy (DCEE) Project

The Topic Modelling research repository for the Digital Circular Electrochemical Economy (DCEE) project at Heriot-Watt University. This research is funded by Digital Circular Electrochemical Economy ([EP/V042432/1](https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/V042432/1)), and the UK Research and Innovation (UKRI) Interdisciplinary Centre for Circular Chemical Economy ([EP/V011863/1](https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/V011863/1)). In response to this call, we have united a cross-disciplinary team of leading researchers from three UK universities: Imperial College London, Loughborough University, and Heriot-Watt University.

### Data and Results

The datasets and experimental results will be made publicly available following the [EPSRC Data Storage Policy](https://www.ukri.org/who-we-are/epsrc/our-policies-and-standards/policy-framework-on-research-data/principles/) and [GDPR Regulations](https://gdpr-info.eu/). Currently, only the code for the models, hyperparameter optimization experiments, and data preprocessing scripts are publicly available. Full datasets and results will be shared after approval.

### Publication

This research has been submitted to the **JCR Q1** Elsevier journal ["Energy and AI"](https://www.sciencedirect.com/journal/energy-and-ai). The preprint is now available on [arXiv](https://arxiv.org/abs/2405.10452).

### How to Use

#### Creating a Python 3.8 Environment

To ensure compatibility with the code, it is recommended to create a Python 3.8 virtual environment. Follow these steps:

##### Option 1: Using virtualenv

1. Install Python 3.8 and virtualenv if you haven't already.
2. Create a virtual environment:
   ```sh
   virtualenv -p python3.8 venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```sh
     venv\Scripts\activate
     ```
   - On Unix or MacOS:
     ```sh
     source venv/bin/activate
     ```
4. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

##### Option 2: Using conda

1. Install Anaconda or Miniconda if you haven't already.
2. Create a conda environment with Python 3.8:
   ```sh
   conda create --name dcee python=3.8
   ```
3. Activate the conda environment:
   ```sh
   conda activate dcee
   ```
4. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

### Running the Scripts

The repository contains scripts for different models (BERTopic, CorEx, LDA) and preprocessing steps. You can find the scripts in the `scripts` directory. Each subdirectory contains Jupyter notebooks (`.ipynb`) and Python scripts (`.py`) for hyperparameter tuning and grid search operations.

To run a specific script, navigate to its directory and execute the script. For example:
```sh
cd scripts/bertopic
python bert_grid_guardian.py
```

### License

This project is licensed under the **Apache 2.0 License** - see the [LICENSE](LICENSE) file for details.

### Contact

For any questions or further information, please contact the project team at [dcee.org.uk](https://dcee.org.uk/).
