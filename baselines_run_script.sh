#!/bin/bash

python_scripts=(
    "models/baselines/no_pretrain/resnet/resnet_train_val.py"
    "models/baselines/no_pretrain/densenet/densenet_train_val.py"
    "models/baselines/no_pretrain/efficientnet/efficientnet_train_val.py"
    "models/baselines/pretrain/resnet/resnet_train_val.py"
    "models/baselines/pretrain/densenet/densenet_train_val.py"
    "models/baselines/pretrain/efficientnet/efficientnet_train_val.py"
)

for script in "${python_scripts[@]}"; do
    echo " "
    echo "Running $script ..."
    time python "$script"
    echo "Completed $script"
    echo " "
done

echo "All Python scripts have been executed."
