#!/bin/bash

GPU="0"
# Specify datasets with time horizon
# datasets=("wikipedia 3600" "reddit 900" "mooc 1200" "lastfm 21600" "myket 5400" "enron 172800" "SocialEvo 1800" "uci 57600" "Flights 1" "CanParl 1" "USLegis 1" "UNtrade 1" "UNvote 1" "Contacts 1")
datasets=("wikipedia 3600" "USLegis 1")
# Specify models
models=("JODIE" "DyRep" "TGN" "TGAT" "CAWN" "EdgeBank" "TCL" "GraphMixer" "DyGFormer")

# Run experiments for each dataset with corresponding horizon and model
for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        # Split the dataset string into an array
        IFS=' ' read -ra dataset_array <<< "$dataset"
        
        # If model is EdgeBank, use evaluation instead of training
        if [ $model == "EdgeBank" ]
        then
            python evaluate_link_prediction.py --dataset_name ${dataset_array[0]} --model_name $model --patience 5 --load_best_configs --num_runs 5 --gpu $GPU --negative_sample_strategy "historical"
        # Else, train the model and evaluate on link prediction
        else
            python train_link_prediction.py --dataset_name ${dataset_array[0]} --model_name $model --patience 5 --load_best_configs --num_runs 5 --gpu $GPU --negative_sample_strategy "historical"
        fi
        # Evaluate the model on link forecasting
        python evaluate_link_prediction.py --dataset_name ${dataset_array[0]} --model_name $model --patience 5 --load_best_configs --num_runs 5 --gpu $GPU --negative_sample_strategy "historical" --horizon ${dataset_array[1]}
    done
done
