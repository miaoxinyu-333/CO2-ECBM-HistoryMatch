# CO2-ECBM-HistoryMatch

**CO2-ECBM-HistoryMatch** is a project focused on history matching and simulation in the fields of CO2 and Enhanced Coal Bed Methane (ECBM) recovery. This project employs deep learning and data assimilation techniques, using the Ensemble Smoother with Multiple Data Assimilation (ESMDA) algorithm. It trains a three-stage forward model, which includes a feature refinement model, a feature reconstruction model (Inverse-PCA), and a surrogate model, to efficiently fit historical data and make predictions.

## Project Structure

```
CO2-ECBM-HistoryMatch
├── config                    # Configuration files and related Python modules
│   ├── FourierModelConfig.py # Python config file for Fourier model
│   ├── FourierUNet.yaml      # YAML config file for Fourier UNet model
│   ├── pca.yaml              # YAML config file for PCA model
│   ├── PCAModelConfig.py     # Python config file for PCA model
│   ├── __init__.py           # Initialization file for the config module
├── data/                     # Data module for handling data loading and preprocessing
│   ├── AHMDataModule.py      # History matching data module, defines the data handling process
│   ├── geo_model.py          # Gaussian distribution of permeability and porosity
│   ├── geo_pycomsol.py       # mph library for comsol 
│   └── __init__.py           # Initialization file for the data module
├── dataSet                   # Directory for raw and reconstructed data
│   ├── processed             # Processed datasets
│   ├── raw                   # Raw datasets (HDF5 format)
├── evaluation                # Evaluation module
│   ├── eval_metric.py        # Code defining model evaluation metrics
│   ├── __init__.py           # Initialization file for the evaluation module
├── logs                      # Stores training logs, model checkpoints, and generated images
│   ├── images                # Stores generated image files
│   ├── pcaModel              # Stores results for the PCA model
│   ├── recModel              # Stores results for the reconstruction model
│   ├── surrogateModel        # Stores results for the surrogate model
│   ├── tb_logs               # TensorBoard log files
├── models                    # Model definition module
│   ├── AHMModel.py           # History matching model definition
│   ├── ForwardModel.py       # Forward model definition
│   ├── PCAModel.py           # PCA model definition
│   ├── __init__.py           # Initialization file for the models module
├── modules                   # Modules defining neural network structures, loss functions, etc.
│   ├── ESMDA.py              # Data assimilation algorithm module
│   ├── FourierUnet.py        # Fourier UNet model definition
│   ├── Unet.py               # UNet model definition
│   ├── Resnet.py             # ResNet model definition
│   ├── __init__.py           # Initialization file for the modules
├── scripts                   # Script files for running various training and assimilation tasks
│   ├── dataAssimilation.py   # Data assimilation script
│   ├── run_all.py            # Script to run all tasks at once
│   ├── __init__.py           # Initialization file for scripts
├── training                  # Directory for training scripts
│   ├── train_autoencoder_per.py  # Permeability autoencoder training script
│   ├── train_autoencoder_por.py  # Porosity autoencoder training script
│   ├── train_PCA.py              # PCA model training script
│   ├── train_surrogateModel.py   # Surrogate model training script
│   ├── __init__.py               # Initialization file for training module
├── utils                     # Utility module containing tools for data handling, evaluation, visualization, etc.
│   ├── dataUtils.py          # Data handling utilities
│   ├── plotUtils.py          # Visualization utilities
│   ├── metricsUtils.py       # Evaluation metrics utilities
│   ├── assimilationUtils.py  # Data assimilation utilities
│   ├── __init__.py           # Initialization file for utils
├── environment.yml           # Conda environment configuration file
├── LICENSE                   # Project license file
├── README.md                 # Project documentation file
├── requirements.txt          # Python dependencies file
```

## Project Modules

The project is divided into the following main modules:

1. **Configuration Module (`config`)**:
    - Stores all configuration files for the models and algorithms, including both YAML config files and Python classes.

2. **Data Module (`data`)**:
    - Responsible for loading and preprocessing datasets, handling various data formats and files used in the history matching process.

3. **Dataset Directory (`dataSet`)**:
    - Stores raw and processed datasets, including permeability, porosity, and other data in HDF5 format.

4. **Evaluation Module (`evaluation`)**:
    - Defines and calculates evaluation metrics for model performance, such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).

5. **Model Module (`models`)**:
    - Contains all model architectures, including autoencoders, UNet, ResNet, and PCA models.

6. **Scripts Module (`scripts`)**:
    - Scripts for running data assimilation, model training, and other tasks. A script is also provided to run all tasks at once.

7. **Training Module (`training`)**:
    - Contains scripts for training models such as permeability and porosity autoencoders, PCA, and surrogate models.

8. **Utilities Module (`utils`)**:
    - Contains utility functions for data handling, visualization, evaluation, and data assimilation.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/miaoxinyu-333/CO2-ECBM-HistoryMatch.git
   ```

2. Install the required dependencies:

   - First, create and activate the `conda` environment:

     ```bash
     conda env create -f environment.yml
     conda activate co2_env
     ```

   - Then, use `pip` to install the additional dependencies listed in `requirements.txt`:

     ```bash
     pip install -r requirements.txt
     ```

## Configuration Files

The project uses YAML configuration files to store training parameters, dataset paths, logging paths, etc. Each task has its own config file located in the `config/` directory, such as `FourierUNet.yaml` and `pca.yaml`.

In each training script, the path to the configuration file is specified like this:

```python
config_path = os.path.join("config", "FourierUNet.yaml")
```

## Data Set

## Logs File 


## Usage

### 1. Train Autoencoder Models

The project includes two autoencoder training scripts, one for the permeability channel and one for the porosity channel.

#### Train the Permeability Autoencoder

Run the following command to start training:

```bash
python training/train_autoencoder_per.py
```

#### Train the Porosity Autoencoder

Run the following command to start training:

```bash
python training/train_autoencoder_por.py
```

### 2. Train PCA Model

Run the following command to train the PCA model:

```bash
python training/train_PCA.py
```

### 3. Train Surrogate Model

Run the following command to train the surrogate model:

```bash
python training/train_surrogateModel.py
```

### 4. Data Assimilation

The project supports data assimilation using the Ensemble Smoother with Multiple Data Assimilation (ESMDA) process.

#### Run Data Assimilation

Use the following command to run the data assimilation script:

```bash
python scripts/dataAssimilation.py
```

The script will:
- Initialize the ESMDA model
- Load observation data
- Run the data assimilation process
- Generate error plots and parameter assimilation results, saved in the `logs/images/assimilation` directory

The output includes the final Mean Squared Error (MSE) and Mean Absolute Error (MAE).

### 5. Run All Tasks

If you want to run all training and data assimilation tasks at once, you can use the `scripts/run_all.py` script.

Run the following command:

```bash
python scripts/run_all.py
```

This script will sequentially run:
- `train_PCA.py`
- `train_autoencoder_per.py`
- `train_autoencoder_por.py`
- `train_surrogateModel.py`
- `dataAssimilation.py`

Results will be logged, and any errors will stop the process.

## Documentation

The documentation for this project is already pre-built and can be viewed easily.

### Viewing Documentation Locally

To view the documentation locally, simply run the following command from the project root directory:

```bash
start build/html/index.html
```

## Logs and Checkpoints

During training, `TensorBoard` is used to log the training progress, and the `ModelCheckpoint` feature is used to save the best-performing model. The paths for logs and checkpoints are specified in the config files.

#### TensorBoard

Start TensorBoard by running:

```bash
tensorboard --logdir=./logs/tb_logs
```

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Project Link

GitHub Repository: [https://github.com/miaoxinyu-333/CO2-ECBM-HistoryMatch](https://github.com/miaoxinyu-333/CO2-ECBM-HistoryMatch)

## Acknowledgements

This project would not have been possible without the contributions of the following open-source projects and repositories. We would like to express our sincere gratitude to the authors and maintainers of these projects for their incredible work:

- [Pdearena](https://github.com/pdearena/pdearena): Special thanks to the pdearena repository for providing the neural network architecture that forms the foundation of our model. Their work on solving PDEs with neural networks was instrumental in the development of this project.
- [PyTorch](https://pytorch.org/): For providing a powerful and flexible deep learning framework used extensively in this project.
- [Scikit-learn](https://scikit-learn.org/): For offering the PCA implementation and other essential machine learning utilities used in model development.
- [COMSOL mph library](https://github.com/MPh-py/MPh): For enabling the integration of COMSOL with Python in the geomechanical simulation tasks.
- [COMSOL](https://www.comsol.com/): For providing the multiphysics simulation platform used for the geomechanical and fluid flow simulations, which are key components of the history matching process.

This work was fully supported by the Anhui Province Science and Technology Major Special Projects (202203a07020010). We would also like to extend our gratitude to the Institutional Center for Shared Technologies and Facilities of INEST,HFIPS,CAS for their technical support.


## Contact

For any inquiries or further information, please feel free to contact:

- **Student Contact**: Xinyu Miao ([1185190409@qq.com](mailto:1185190409@qq.com))
- **Mentor Contact**: [Chunhua Chen] ([mentor_email@example.com](mailto:mentor_email@example.com))

If you have any questions regarding the project or need additional information, please reach out to the student. For more detailed technical or supervisory queries, you can contact the project mentor directly.