# Executorch

## Pre-requisites:

- Confirm you have the model 36 weights (might be called best_metric_model.pth or whatever Chris sent you)

## Steps

### Step 1: Clone the Executorch repo

```sh
cd /path/to/project

git clone https://github.com/pytorch/executorch.git
```

### Step 2: Set up python environment

Can use conda or anything, I'll use virtualenv venv

```sh
python3 -m venv executorch_venv

source executorch_venv/bin/activate

# Enter the executorch repo
cd executorch

# Update and pull submodules
git submodule sync
git submodule update --init

# Install ExecuTorch pip package and its dependencies, as well as
# development tools like CMake.
# If developing on a Mac, make sure to install the Xcode Command Line Tools first.
# Use xnnpack for example
./install_requirements.sh --pybind xnnpack

# Return to original repo because need requirements on our end
cd ..

pip install -r requirements_executorch.txt
```

### Step 3. Use python to export the file as Executorch program

See the notebook: [model_to_executorch_quantized.ipynb](./model_to_executorch_quantized.ipynb)

This will create a .pte file (model_threesix_quantized.pte) which we use in later steps

### Step 4. Run the cmake commands to run locally with XNN

To test if executorch conversion worked, do the following:

I write the bash script to run it for you. Run it from the export_executorch folder:

```sh
cd export_executorch_quantized
bash build_xnn.sh
```

To see the outputs, run:

```sh
./cmake-out/backends/xnnpack/xnn_executor_runner --model_path=./model_threesix_quantized.pte
```

You should see: Output 0: tensor(sizes=[1, 3], [0., 1.83687, 0.706489]) with maybe slightly different values (because calibration with random numbers might be different)

For example in the model provided [model_threesix_quantized.pte](./model_threesix_quantized.pte) the output is: Output 0: tensor(sizes=[1, 3], [0., 1.10635, 2.48313])

The descrepancy in values can be explained by the strange inputs (all ones) or the fact the calibration (see middle steps in the notebook) is done with random normal distributed values for images which is only quasi-realistic. Real cervical images should be used to calibrate in the future.

### Step 5. (Optional) Run on android simulator

To run on Android device:

Untested but see this step in [export_executorch README](../export_executorch/README.md#step-5-optional-run-on-android-simulator)