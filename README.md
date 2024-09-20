# Model Conversion

Converting PyTorch models to Android devices

## Overview

The repo has 3 modules:

- Export TFLite (export_tflite)
- Export Executorch (export_executorch)
- Export Executorch with dropout (export_executorch_with_dropout)

Each module has its own README.md file which gives the step-by-step breakdown how to convert the model, and then ultimately run inference once to see it worked. Both TFLite and Executorch modules as self contained and can be run from scratch. Executorch with dropout only has one slight difference which changes 2 steps. Therefore they can be run in any order but for executorch probably do without dropout first.

I include the weights for model 36 since they are small enough to fit in repo and can serve as backup if Chris hasn't sent more recent ones.

If there are any questions please email me at scott.kinder@cuanschutz.edu