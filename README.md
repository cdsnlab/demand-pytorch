# Demand-PyTorch Execution Manual

---

### I. Introduction: Overview of Demand-PyTorch Framework

The Demand-PyTorch framework offers a suite of advanced deep learning models for demand forecasting, such as ConvLSTM, DMVSTNet, and STMetaNet. It supports diverse urban scenarios like bike-sharing and public transit. This manual serves as a comprehensive guide to execute and customize the framework.

---

### II. Environment Setup and Requirements

1. **System Requirements**:
    - Operating System: Linux, macOS, or Windows with WSL support.
    - Python Version: 3.8 or higher.
    - Memory: Minimum 8GB RAM (16GB recommended).
    - GPU: NVIDIA GPU with CUDA support.

2. **Dependency Installation**:
    - Clone the repository and install dependencies:
      ```bash
      git clone https://github.com/your-repo/demand-pytorch.git
      cd demand-pytorch
      pip install -r requirements.txt
      ```

3. **Data Preparation**:
    - Ensure datasets are preprocessed. Utilities like `UrbanSTC_data_process.py` or `datasets.py` can help convert raw data into compatible formats.
    - Preprocessed datasets for Beijing or NYC are included in the repository.

---

### III. Project Structure

1. **Key Directories**:
    - `config`: Model-specific configurations (ConvLSTM, DMVSTNet, etc.).
    - `data`: Dataset handling utilities.
    - `evaluation`: Metrics such as MAE, RMSE, and MAPE.
    - `logger`: Tools for logging experiment details.
    - `model`: Implementations of forecasting models.
    - `trainer`: Training scripts for each model.
    - `util`: Supporting scripts for data preprocessing and logging.

2. **Important Files**:
    - `train.py`: Main script for training.
    - `config/*.py`: Configuration files for each model.
    - `trainer/base_trainer.py`: Base trainer for extending training logic.

---

### IV. Execution Steps

1. **Configuring the Model**:
    - Select the model by editing its configuration file in `config/`.
    - Example: Modify learning rates or input shapes in `STResNet_config.py`.

2. **Training the Model**:
    - Use `train.py` to start the training process:
      ```bash
      python train.py --model STResNet --config config/STResNet_config.py
      ```

3. **Evaluating the Model**:
    - Compute evaluation metrics using `evaluation/metrics.py`:
      ```bash
      python evaluation/metrics.py --predictions results/pred.csv --ground_truth data/ground_truth.csv
      ```

4. **Visualizing Results**:
    - Generate visualizations of results using the `vis` directory tools:
      ```bash
      python vis/buildgif.py --data results/predictions.csv
      ```

---

### V. Notes

1. **Custom Dataset Integration**:
    - Use `data/datasets.py` or `UrbanSTC_data_process.py` for preprocessing new datasets.
    - Ensure compatibility with `.csv` or `.pkl` formats.

2. **Extending Models**:
    - Add custom models in the `model/` directory.
    - Follow the structure of existing models like `STMGCN.py` or `DeepSTN.py`.

3. **Error Debugging**:
    - Logs from experiments are available in the specified log directory in the configuration files.

---

### VI. Conclusion

Demand-PyTorch provides a flexible platform for demand forecasting with multiple state-of-the-art models. By adhering to this manual, users can train, evaluate, and extend models to address diverse urban demand forecasting challenges.


---

# demand-pytorch 

## Results

![bj-flow](demand-pytorch/vis/bj.gif)

We report MAE in Beijing taxi flow prediction dataset for each prediction steps.
| Model | bj-flow (15 min) | bj-flow  (30 min) | bj-flow (1 hour) |
|-------|--|--|--|
| ConvLSTM | | | | |
| STResNet | | | | |
| DeepSTN  | | | | |

We report RMSE and MAPE in NYC taxi flow prediction dataset.
| Model | RMSE | MAPE | 
|-------|--|--|
| DMVST-Net | |

## Getting Started

### Data
- Beijing Taxi: download bj-flow.pickle from [Google Drive]()
- NYC Taxi:

### Environment
``` 
conda create -n $ENV_NAME$ python=3.7
conda activate $ENV_NAME$

# CUDA 11.3
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 
# Or, CUDA 10.2 
pip install torch==1.11.0+cu102 --extra-index-url https://download.pytorch.org/whl/cu102 
pip install -r requirements.txt
```

### Train
If config file not specified, load $MODEL_NAME$_config.py by default. 
```
python train.py --model $MODEL_NAME$ --ddir $PATH_TO_DATASET$ --dname $DATASET_NAME$
```

