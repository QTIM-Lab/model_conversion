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

pip install -r requirements.txt
```

### Step 3. Use python to export the file as Executorch program

See the notebook: [model_to_executorch.ipynb](./model_to_executorch.ipynb)

This will create a .pte file (model_threesix.pte) which we use in later steps

### Step 4. Run the cmake commands to run locally

To test if executorch conversion worked, do the following:

I write the bash script to run it for you. Run it from the export_executorch folder:

```sh
cd export_executorch
bash build_test.sh
```

To see the outputs, run:

```sh
./cmake-out/executor_runner --model_path model_threesix.pte
```

You should see: Output 0: tensor(sizes=[1, 3], [-2.49069, 1.39235, 0.966962])

### Step 5. (Optional) Run on android simulator

To run on Android device:

First make sure you have the NDK: [https://developer.android.com/ndk/downloads](https://developer.android.com/ndk/downloads)

I use r27b and I believe that is recommended but probably okay to use other version

Then run the bash script I wrote:

```sh
bash build_xnn_android.sh
```

That will create a file: cmake-android-out/extension/android/libexecutorch_jni.so

And that file is what should be served by the android app.

Then finally if you want to deploy on an example app see this example here: [https://pytorch.org/executorch/stable/demo-apps-android.html](https://pytorch.org/executorch/stable/demo-apps-android.html) and we are at the "Deploying on Device via Demo App" step with model 36. The demo app would then need to have code adjusted to do anything useful wuth model 36 as it currently expects a different kind of model.