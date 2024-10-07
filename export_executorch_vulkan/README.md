# Executorch with Vulkan

## Pre-requisites:

- Confirm you have the model 36 weights (might be called best_metric_model.pth or whatever Chris sent you)

## Steps

(same as others up to step 3)

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

See the notebook: [model_to_executorch_vulkan.ipynb](./model_to_executorch_vulkan.ipynb)

This will create a .pte file (model_threesix_vulkan.pte) which we use in later steps

### Step 4. Build and install ExecuTorch libraries with Vulkan Delegate

First, make sure that you have the Android NDK installed - Android NDK r25c is recommended, but r27b worked for me. The Android SDK should also be installed so that you have access to adb.

Link to NDK download: [https://developer.android.com/ndk/downloads](https://developer.android.com/ndk/downloads)

```sh
# Make sure adb works
adb --version
```

Now that we have the model, we need to be able to run it

I wrote a bash script. Run it from this export_executorch_vulkan folder:

First build and install:

```sh
cd export_executorch_vulkan
bash build_vulkan_android.sh
```

### Step 5. Push model and binary to device and run

Next, you must have device visible to adb. For instance, if you run:

```sh
# Check if devices are present
adb device
```

You should see at least one device. For instance if I have Android Studio running and have some emulator (say Pixel 7 on x86_64 or whatever), then that will work

Then, to see the outputs, run:

```sh
# Push model to device
adb push model_threesix_vulkan.pte /data/local/tmp/model_threesix_vulkan.pte
# Push binary to device
adb push cmake-android-out/backends/vulkan/vulkan_executor_runner /data/local/tmp/runner_bin

# Run the model
adb shell /data/local/tmp/runner_bin --model_path /data/local/tmp/model_threesix_vulkan.pte
```

You should see: Output 0: tensor(sizes=[1, 3], [-2.49069, 1.39235, 0.966965])

Which is exactly the same except I think in other versions it is: Output 0: tensor(sizes=[1, 3], [-2.49069, 1.39235, 0.966962]). But logits within 0.00001 is fine precision
