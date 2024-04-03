```markdown
# Autify_ML_Assignment

This repository contains the dataset, code for models, and model weights for the
Autify, Inc. technical assignment for the Senior Machine Learning Engineer, LLMs & Prompt Engineering (New Project) position.

## Overview

PLEASE NOTE: read the autify_challenge.pdf document for an overview, approach, thought process, analysis, problem-solving, and conclusion for this challenge.
The approach taken involves a hybrid approach of both research and deployment.

## Dataset

The `checkbox_state_v2` directory contains the training and inference data.
The training data is in the original format and can be used directly.
For inference, new images need to be added to the `inference` folder for testing.
PLEASE NOTE: For inference, only one image at a time should be added for testing due to time constraints I have not implemented warning handling and multi image processing and other quality of life features.
However, this feature can be easily extended to handle multiple images later on.
I understand that this could be an inconvenience during testing and I apologise for the same.

## Model Weights

The `model_weights` folder contains the model weights of all the models.
The respective file paths are set to load or save the weights.
PLEASE NOTE: retraining the models will replace the old weights in the folder with the new ones.
Make sure to have a backup copy.

## Model Code

The `models` folder contains the code for all the baselines and checkpoints.
The main folder contains the final model code used for inference.
PLEASE NOTE: For more details and explanations of baselines, checkpoints, and final model files, please refer to the autify_challenge.pdf document.

## Setup Instructions

To set up the environment, follow these steps:

1. Create a Python virtual environment:
   ```
   python -m venv autify_challenge
   ```

2. Activate the virtual environment:
   ```
   source autify_challenge/bin/activate
   ```

3. Install necessary libraries from requirements.txt:
   ```
   pip install -r requirements.txt
   ```

Note: This project uses matplotlib visualizations for testing files only.
If tkinter is not installed globally, it might throw an error.
Install tk using one of the following commands based on your system:

- Arch Linux: `pacman -S tk`
- Debian/Ubuntu: `apt install python3-tk`
- Fedora: `dnf install python3-tkinter`
- macOS: `brew install python-tk`
- Windows: ToDo

## Usage

- To view test results of the final model, run:
  ```
  python main/efficientnet_test.py
  ```

- To submit your images for inference, copy the image one at a time to the `inference/data` folder inside checkbox_state_v2 and then run:
  ```
  python main/efficientnet_inference.py
  ```

- To train the model again, run:
  ```
  python main/efficientnet_train_val.py
  ```

- To visualize test results and accuracy for baseline and checkpoint models, run the following commands:
  ```
  python models/baselines/no_pretrain/resnet/resnet_test.py
  
  python models/baselines/no_pretrain/densenet/densenet_test.py
  
  python models/baselines/no_pretrain/efficientnet/efficientnet_test.py
  
  python models/baselines/pretrain/resnet/resnet_test.py
  
  python models/baselines/pretrain/densenet/densenet_test.py
  
  python models/baselines/pretrain/efficientnet/efficientnet_test.py
  
  python models/checkpoints/checkpoint_1/efficientnet_test.py
  
  python models/checkpoints/checkpoint_2/efficientnet_test.py
  
  python models/checkpoints/checkpoint_3/efficientnet_test.py
  
  python models/checkpoints/checkpoint_4/efficientnet_test.py
  
  python models/checkpoints/checkpoint_5/efficientnet_test.py
  ```

- If you want to train all the models from scratch, refer and run the following files after making them executable:
  ```
  ./baselines_run_script.sh
  
  ./checkpoints_run_script.sh
  ```

Please ensure file permissions are set correctly before running scripts.

## Note

Feel free to explore the codebase and provide any feedback or improvements or ask any questions.
Again due to time constraints I have not implemented the Dockerfile as mentioned in the project please let me know if you would like to see this as well at a later date.
```
