# TFLite

## Pre-requisites:

- Confirm you have the model 36 weights (might be called best_metric_model.pth or whatever Chris sent you)

## Steps

### Step 1. Set up python environment

```sh
python3 -m venv tflite_venv
pip install -r requirements_tflite.txt
```

### Step 2. Convert PyTorch model to TFLite

See the notebook: [model_to_tflite.ipynb](./model_to_tflite.ipynb)

This will create a .tflite file (model_thirtysix.tflite) which we use in later steps. You can also confirm the values of the output match.

### Step 3. Run in TFLite Runtime

To check output is correct, I have an example TFLite interpreter python script:

```sh
python run_interpreter.py
```

The output should be:

Output data shape: (1, 3)

Output data:  [ 8.182903  -2.287459  -6.3397408]

