# Executorch with Dropout

## Pre-requisites:

- Confirm you have the model 36 weights (might be called best_metric_model.pth or whatever Chris sent you)

## Steps

### Step 1-2: Clone the Executorch repo and Python environment

Same as from [export_executorch](../export_executorch/README.md).

### Step 3. Use python to export the file as Executorch program

See the notebook: [model_to_executorch_with_dropout.ipynb](./model_to_executorch_with_dropout.ipynb)

This will create a .pte file (model_threesix.pte) which we use in later steps

### Step 3.5 Copy the Dropout C++ script to the proper location

Before we build with cmake, which looks for the underlying implementations (which need to be custom written for mobile devices, but most have been written by Torch team besides a few, like dropout) we need to copy the Dropout script over. We also need to copy a yaml file that the builder uses.

```sh
cp ./op_native_dropout.cpp ../executorch/kernels/portable/cpu
cp ./functions.yaml ../executorch/kernels/portable
```

If you are curious you can look around at ../executorch/kernels/portable/cpu for all the other implementations of the C++ code for the deep learning operations like ReLU, Conv2d, etc.

Now that it is in there, the following steps will work the same as step 4 and onwards at [export_executorch](../export_executorch/README.md). But I will still copy and paste the steps with the small changes that are now present. And if you are curious, if we didn't have this C++ script (or the yaml with the changes for Dropout), it would say the Dropout op is not implemented and not return an output when we try to run

### Step 4. Run the cmake commands to run locally

To test if executorch conversion worked, do the following:

I write the bash script to run it for you. Run it from the export_executorch folder:

```sh
bash build_test.sh
```

To see the outputs, run:

```sh
./cmake-out/executor_runner --model_path model_threesix_dropout.pte
```

You should see something like: Output 0: tensor(sizes=[1, 3], [-2.27883, 2.32634, -2.0448])

But because there is dropout, the values should be different. Run it twice to make sure the values are indeed different each time.

### Step 5. (Optional) Run on android simulator

Same as [export_executorch](../export_executorch/README.md).