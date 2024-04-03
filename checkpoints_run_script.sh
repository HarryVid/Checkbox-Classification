#!/bin/bash

python_scripts=(
    "models/checkpoints/checkpoint_1/efficientnet_train_val.py"
    "models/checkpoints/checkpoint_2/efficientnet_train_val.py"
    "models/checkpoints/checkpoint_3/efficientnet_train_val.py"
    "models/checkpoints/checkpoint_4/efficientnet_train_val.py"
    "models/checkpoints/checkpoint_5/efficientnet_train_val.py"
)

for script in "${python_scripts[@]}"; do
    echo " "
    echo "Running $script ..."
    time python "$script"
    echo "Completed $script"
    echo " "
done

echo "All Python scripts have been executed."
