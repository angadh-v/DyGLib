#!/bin/bash

GPU="0"
# Specify datasets with time horizon
datasets=("wikipedia 3600")
# Specify models
# models=("JODIE" "DyRep" "EdgeBank" "DyGFormer")
models=("EdgeBank")
# Run experiments for each dataset with corresponding horizon and model
for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        # Split the dataset string into an array
        IFS=' ' read -ra dataset_array <<< "$dataset"
        
        # Since EdgeBank is not neural network-based, it does not need training
        if [ $model != "EdgeBank" ]
        then
            # echo "skip"
            python train_link_prediction.py --dataset_name ${dataset_array[0]} --model_name $model --patience 5 --load_best_configs --num_runs 5 --gpu $GPU --negative_sample_strategy "historical"
        fi
        # Evaluate model on link prediction
        python evaluate_link_prediction.py --dataset_name ${dataset_array[0]} --model_name $model --patience 5 --load_best_configs --num_runs 5 --gpu $GPU --negative_sample_strategy "historical"
        # Evaluate model on link forecasting
        python evaluate_link_prediction.py --dataset_name ${dataset_array[0]} --model_name $model --patience 5 --load_best_configs --num_runs 5 --gpu $GPU --negative_sample_strategy "historical" --horizon ${dataset_array[1]}
    done
done
