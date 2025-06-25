### 🌍🔍 Topic Modelling Research in the Digital Circular Electrochemical Economy (DCEE) Project

The Topic Modelling research repository for the Digital Circular Electrochemical Economy (DCEE) project at Heriot-Watt University. This research is funded by Digital Circular Electrochemical Economy ([EP/V042432/1](https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/V042432/1)), and the UK Research and Innovation (UKRI) Interdisciplinary Centre for Circular Chemical Economy ([EP/V011863/1](https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/V011863/1) and [EP/V011863/2](https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/V011863/2)). In response to this call, we have united a cross-disciplinary team of leading researchers from three UK universities: Imperial College London, Loughborough University, and Heriot-Watt University.

### 📊 Data and Results

The main dataset for this project is now publicly available via the university's open access repository:

- **DOI:** [10.17861/85bf3f9d-dc42-4b5c-8e29-47ddd0f0f687](https://doi.org/10.17861/85bf3f9d-dc42-4b5c-8e29-47ddd0f0f687)
- **License:** This dataset is released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license.

The datasets and experimental results are made publicly available following the 🔐 [EPSRC Data Storage Policy](https://www.ukri.org/who-we-are/epsrc/our-policies-and-standards/policy-framework-on-research-data/principles/) and 📜 [GDPR Regulations](https://gdpr-info.eu/).

### 🏆 Publication

🎊 The [paper](https://www.sciencedirect.com/science/article/pii/S2666546824000995) has been published in the **JCR Q1** Elsevier journal '**[Energy and AI](https://www.sciencedirect.com/journal/energy-and-ai)**' 🎉. 

🔥 The [preprint](https://arxiv.org/abs/2405.10452) is available on arXiv 🚀.

### ⚙️ How to Use

#### Creating a 🐍 Python 3.8 Environment

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

### 🚀 Running the Scripts

The repository contains scripts for different models (BERTopic, CorEx, LDA) and preprocessing steps. You can find the scripts in the `scripts` directory. Each subdirectory contains Jupyter notebooks (`.ipynb`) and Python scripts (`.py`) for Single-objective Optimisation and BERTopic contains Single and Multi-objective Optimisation.

To run a specific script, navigate to its directory and execute the script. For example:
```sh
cd scripts/bertopic
python bert_grid_guardian.py
```

### 📜 License

This project is licensed under the **Apache 2.0 License** - see the [LICENSE](LICENSE) file for details.

**Dataset License:** The dataset is released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license.

### 📬 Contact

For any questions or further information, please contact the project team at 🌐 [Digital Circular Electrochemical Economy (DCEE) Project](https://dcee.org.uk/) and 🏛️ [National Interdisciplinary Centre for the Circular Chemical Economy](https://www.circular-chemical.org/).

---

### 🔖 Citation

**If you use our code or refer to our publication, please cite the following BibTeX:**

```bibtex
@article{song2024exploring,
  title={Exploring public attention in the circular economy through topic modelling with twin hyperparameter optimisation},
  author={Song, Junhao and Yuan, Yingfang and Chang, Kaiwen and Xu, Bing and Xuan, Jin and Pang, Wei},
  journal={Energy and AI},
  pages={100433},
  year={2024},
  publisher={Elsevier}
}
```

If you use our dataset, please cite the following entry in your BibTeX:

```bibtex
@dataset{song2025public,
  author    = {Song, Junhao and Yuan, Yingfang and Chang, Kaiwen and Xu, Bing and Xuan, Jin and Pang, Wei},
  title     = {Public Attention Text Dataset on Circular Economy for Topic Modelling},
  year      = {2025},
  publisher = {Heriot-Watt University},
  doi       = {10.17861/85bf3f9d-dc42-4b5c-8e29-47ddd0f0f687},
  url       = {https://doi.org/10.17861/85bf3f9d-dc42-4b5c-8e29-47ddd0f0f687},
  note      = {EAI2024Data(.zip)}
}
```