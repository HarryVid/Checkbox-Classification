# Autify_ML_Assignment
Autify, Inc. | technical assignment for the Senior Machine Learning Engineer, LLMs &amp; Prompt Engineering (New Project) position! [Hiring 2022] ML Take-Home Assignment

This GitHub repo contains the dataset, the code for the models and the model_weights for the respective models for this challenge.

Please do not disturb any file or folder structure.

Also, I request you to please read and go through the autify_challenge.pdf document to under my overview, approach, thought process, analysis, problem-solving and conclusion for this challenge, I have taken a hybrid approach of both research and deployment.

The checkbox_state_v2 contains the training and inference data. The training data is in the original format and can be used directly while in the inference folder, new images need to be added for testing.
PLEASE NOTE that for inference only added 1 image at a time for testing because I did not have enough time to handle exceptions or implement multiple-image inferencing properly. I understand this could be an inconvenience and I apologise for that but this feature can easily be added later on.

The model_weights folder contains the model weights of all the models. They need not be changed as the respective file has the respective paths to the weights for either saving or loading.
PLEASE NOTE that retraining the models will replace the old weights in the folder with the new so be careful with that. Either take a backup of the folder or re-download it from GitHub.

The model's folder contains the code for all the baselines and checkpoints. If needed they can be opened up and analyzed the folder structure is straightforward. The main folder contains the final model code which will be used for inference. For more details and further explanations for baselines, checkpoints and final model files please refer to the autify_challenge.pdf document.

To get everything setup:

The first step would be to create a Python virtual environment: python -m venv autify_challenge

Activate the virtual environment: source autify_challenge/bin/activate

Install the necessary libraries provided in the requirements.txt file: pip install -r requirements.txt

This project uses matplotlib visualizations for the testing files only so in some cases it might throw an error if tkinter is not installed globally.

So please install tk using either:

pacman -S tk (Arch)

apt install python3-tk (Debian)

dnf install python3-tkinter (Fedora)

brew install python-tk (mac)

To view the test results of the main/final model run the code: python main/efficientnet_test.py

To submit your images copy the image one at a time to the inference folder mentioned above and then run: python main/efficientnet_inference.py

In case you want to train the model again then run the command: python main/efficientnet_train_val.py

If you want to visualize the test results and accuracy for the baseline and checkpoint models run the following commands:

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

In case you want to train all the models from scratch then refer and run the following files after a chmod +x:

./baselines_run_script.sh

./checkpoints_run_script.sh
