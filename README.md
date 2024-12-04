# CO2-ECBM-HistoryMatch

**CO2-ECBM-HistoryMatch** is a project focused on history matching and simulation in the fields of CO2 and Enhanced Coal Bed Methane (ECBM) recovery. This project employs deep learning and data assimilation techniques, using the Ensemble Smoother with Multiple Data Assimilation (ESMDA) algorithm. It trains a forward model, which includes a Autoencdoerkl model and a Fourier-UNet model, to efficiently fit historical data and make predictions.

## Project Structure

```
CO2-ECBM-HistoryMatch
├── checkModel
│   ├── checkAutoencoder.py
│   ├── checkFourierUNet.py
│   ├── checkUNet.py
│   └── timeFourierUNet.py
├── config
│   ├── SurrogateModel.yaml
│   └── __init__.py
├── data
│   ├── AHMDataModule.py
│   └── __init__.py
├── data_assimilation
│   ├── dataAssimilation.py
│   └── da_result.txt
├── environment.yml
├── evaluation
│   ├── eval_metric.py
│   └── __init__.py
├── geomodel
│   ├── checkh5.py
│   ├── execute.py
│   ├── mph_execute.py
│   ├── surrogate_dataset_create.py
│   └── utils.py
├── LICENSE
├── models
│   ├── AutoencoderKLWrapper.py
│   ├── ForwardModel.py
│   ├── ForwardModelCreator.py
│   ├── ForwardModelCreatorPCA.py
│   ├── ForwardModelPCA.py
│   ├── LossFactory.py
│   ├── lr_scheduler.py
│   ├── ModelFactory.py
│   ├── PCAModel.py
│   ├── SurrogateModel.py
│   └── __init__.py
├── modules
│   ├── activations.py
│   ├── blocks.py
│   ├── ConvLSTM.py
│   ├── ESMDA.py
│   ├── fourier.py
│   ├── FourierUnet.py
│   ├── loss.py
│   ├── SimpleCNN.py
│   ├── Unetbase.py
│   └── __init__.py
├── README.md
├── requirements.txt
├── scripts
│   ├── latents_static.py
│   └── __init__.py
├── sensitivity_analysis
│   ├── parameter_sensitivity.py
│   ├── parameter_sensitivity_analysis.py
│   └── sensititivity_result.txt
├── testForward
│   ├── forwardresult.txt
│   ├── testforward.py
│   └── testForwardPCA.py
├── training
│   ├── train_GeoParameterization_autoencoderkl.py
│   ├── train_GeoParameterization_PCAbase.py
│   ├── train_surrogateModel_cnn.py
│   ├── train_surrogateModel_FourierUNet.py
│   ├── train_surrogateModel_nuetbase.py
│   ├── train_surrogateModel_resnet.py
│   └── __init__.py
├── uncertainty_analysis
│   ├── parameter_uncertainy.py
│   ├── uncertaintyresult.txt
│   └── uncertainty_analysis_results
└── utils
    ├── assimilationUtils.py
    ├── dataUtils.py
    ├── fileUtils.py
    ├── metricsUtils.py
    ├── plotUtils.py
    └── __init__.py
```

## Project Modules

The project is divided into the following main modules:

### 1. Configuration Module (`config`)
- Stores configuration files for the surrogate models and algorithms, including YAML config files like `SurrogateModel.yaml` and an initialization file for easier access.

### 2. Data Module (`data`)
- Responsible for loading and preprocessing datasets, as defined in `AHMDataModule.py`. This module is crucial for preparing data for assimilation and model training.

### 3. Data Assimilation Module (`data_assimilation`)
- Contains scripts for performing data assimilation tasks, including `dataAssimilation.py`.
- Results from assimilation tasks are logged in `da_result.txt`.

### 4. Evaluation Module (`evaluation`)
- Defines evaluation metrics for assessing model performance, with functionality implemented in `eval_metric.py`.
- Common metrics like Mean Squared Error (MSE) and other custom metrics are handled here.

### 5. Geological Model Module (`geomodel`)
- Contains scripts for generating and manipulating geological models:
  - `execute.py` for running simulations.
  - `surrogate_dataset_create.py` for dataset preparation.
- Utilities for handling HDF5 files are included.

### 6. Model Module (`models`)
- Contains implementations of various model architectures:
  - Autoencoders
  - UNet
  - PCA-based models
- Includes factory classes (`ModelFactory.py`) for streamlined model creation and custom loss functions.

### 7. Modules Directory (`modules`)
- Contains core building blocks for model architectures, including:
  - Activation functions
  - Layers
  - Advanced modules like Fourier transformations (`fourier.py`) and ConvLSTM (`ConvLSTM.py`).

### 8. Scripts Module (`scripts`)
- Provides scripts for specific tasks such as generating latent variable visualizations with `latents_static.py`.

### 9. Training Module (`training`)
- Includes scripts for training different models:
  - Autoencoders
  - PCA-based models
  - Surrogate models like Fourier UNet (`train_surrogateModel_FourierUNet.py`).

### 10. Sensitivity Analysis Module (`sensitivity_analysis`)
- Contains scripts for sensitivity analysis, such as `parameter_sensitivity.py`.
- Logs results in `sensititivity_result.txt`.

### 11. Uncertainty Analysis Module (`uncertainty_analysis`)
- Handles uncertainty analysis.
- Results are stored in `uncertaintyresult.txt` and detailed logs in the `uncertainty_analysis_results` directory.

### 12. Test Forward Module (`testForward`)
- Provides testing scripts for forward modeling approaches:
  - `testforward.py` for standard forward models.
  - `testForwardPCA.py` for PCA-based forward models.
- Results are logged in `forwardresult.txt`.

### 13. Utilities Module (`utils`)
- Provides utility functions for:
  - File handling
  - Plotting
  - Data assimilation
  - Metrics computation
- Key utilities include `plotUtils.py` for visualization and `dataUtils.py` for data processing.

### 14. Environment and Dependencies
- The `environment.yml` and `requirements.txt` files define the required dependencies and environments for the project.

---

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

The project uses YAML configuration files to store training parameters, dataset paths, logging paths, etc. Each task has its own config file located in the `config/` directory, such as `SurrogateModel.yaml`.

## DataSet and Logs

Due to the large size of the dataset and log files, they are not included in this repository. You can download both the dataset and logs from the following Google Drive link:

[Download Data Set and Logs from Google Drive](https://drive.google.com/drive/folders/1V4fso84Mc_fLaKv9xpU6hfLItr41IYSN?usp=sharing)

### Instructions:
- After downloading the dataset, place it in the `dataSet/` directory.
- After downloading the logs, place them in the `logs/` directory.

This will ensure that all files are properly located when running the project.

## Usage

### 1. Train Autoencoder Models

The project includes an autoencoder training script that uses KL Divergence regularization for permeability and porosity tensors.

#### Train the GeoParameterization AutoencoderKL

Run the following command to start training:

```bash
python training/train_GeoParameterization_autoencoderkl.py
```

### 2. Train PCA Model

Run the following command to train the PCA model:

```bash
python training/train_GeoParameterization_PCAbase.py
```


### 3. Train Surrogate Model

Run the following command to train the Surrogate model:

```bash
python training/train_surrogateModel_FourierUNet.py
```


```bash
python training/train_surrogateModel_nuetbase.py
```

### 4. Data Assimilation

The project supports data assimilation using the Ensemble Smoother with Multiple Data Assimilation (ESMDA) process.

#### Run Data Assimilation

Use the following command to run the data assimilation script:

```bash
python data_assimilation/dataAssimilation.py
```

The script will:
- Initialize the ESMDA model
- Load observation data
- Run the data assimilation process

### 5. Sensitivity Analysis

```bash
python sensitivity_analysis/parameter_sensitivity.py
```

### 6. Uncertainty Analysis


```bash
python uncertainty_analysis/parameter_uncertainy.py
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
- **Mentor Contact**: [Chunhua Chen] ([chunhua.chen@inest.cas.cn](mailto:chunhua.chen@inest.cas.cn))

If you have any questions regarding the project or need additional information, please reach out to the student. For more detailed technical or supervisory queries, you can contact the project mentor directly.