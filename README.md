# Deep Natural Language Feature Learning for Interpretable Prediction - Repository

This is the official repo for **Deep Natural Language Feature Learning for Interpretable Prediction** @ EMNLP 2023.

## One-line for non-expert

```
python step0_hftest.py --model <your_model> --device <your_device>
```

## Step-by-step

### Step 1: Zero-shot Sub-task Labelisation

```
python 01_one_line.py \
    --api_key <your_api_key_value> \
    --file_name_dict_bsqs <your_file_name_dict_bsqs_value> \
    --file_name_data_train <your_file_name_data_train_value> \
    --sentence_col_name <your_sentence_col_name_value> \
    --sample_size <your_sample_size_value> \
    --seed <your_seed_value> \
    --root_labels <your_root_labels_value> \
    --temp <your_temperature_value> \
    --max_t <your_max_tokens_value> \
    --verbose <your_verbose_value>
```

### Step 2: Training of NLLF Generator

```
python 02_one_line.py \
    --file_name_dict_bsqs <your_file_name_dict_bsqs_value> \
    --root_labels <your_root_labels_value> \
    --sentence_col_name <your_sentence_col_name_value> \
    --model_name <your_model_name_value> \
    --maxlen_s <your_max_token_length_s_value> \
    --maxlen_bsq <your_max_token_length_bsq_value> \
    --batch_size <your_batch_size_value> \
    --epochs <your_epochs_value> \
    --lr <your_learning_rate_value> \
    --verbose <your_verbose_value>\
    --hf_token <your_huggingface_token_value> \
    --repo_name <your_repository_name_value> \
    --username <your_huggingface_username_value>
```

### Step 3.1: NLLF Generation

```
python 03_1_one_line.py \
    --file_name_new_dict_bsqs <your_file_name_new_dict_bsqs_value> \
    --maxlen_s <your_max_token_length_s_value> \
    --maxlen_bsq <your_max_token_length_bsq_value> \
    --username <your_huggingface_username_value> \
    --repo_name <your_repository_name_value> \
    --file_name_data_train <your_file_name_data_train_value> \
    --file_name_data_val <your_file_name_data_val_value> \
    --file_name_data_test <your_file_name_data_test_value> \
    --sentence_col_name <your_sentence_col_name_value> \
    --root_labels <your_root_labels_value> \
    --verbose <your_verbose_value>
```

### Step 3.2: NLLF Integration

```
python step0_hftest.py --model <your_model> --device <your_device>
```

## Experiments




## Citation

If you find this repo useful, please cite our paper:
```
@article{nllf2023,
  title={Deep Natural Language Feature Learning for Interpretable Prediction},
  author={Anonymous EMNLP submission},
  year={2023}
}