## Deep Natural Language Feature Learning for Interpretable Prediction - Repository

This is the official repo for **Deep Natural Language Feature Learning for Interpretable Prediction** @ EMNLP 2023.

> **Figure 1:** Full process of subtask labelisation, NLLFG training, NLLF generation and integration.
<img src="https://docs.google.com/drawings/d/e/2PACX-1vQu6AiBe7sxr-00IgmU8Q9RsqkjJAyQLqYoGKINinlcNftcOSRvdp6rlgdHoXVuxLiuF92-FrD_SiU8/pub?w=2240&h=804" width="980">

## Quick Usage

Welcome to the Natural Language Learned Feature (NLLF) pipeline documentation. This guide provides a comprehensive overview of the parameters and steps required to leverage the power of the NLLF pipeline for your tasks.

#### Parameters Description

This section provides detailed descriptions of the variables used in the NLLF pipeline across different steps (Figure 1) of the process.

- **api_key**: Stores the OpenAI API key, necessary for authenticating and accessing the GPT-3.5-turbo language model. This key enables communication with OpenAI's services to perform weak labeling tasks.

- **file_name_dict_bsqs**: Represents the name of the JSON file containing Binary SubTask Questions (BSQ) used in the weak labeling strategy. These questions are formulated to generate weak labels for sentences in the dataset. Additionally, in Step 2, this file serves as references to load pre-generated weak labels. These questions guide the loading of weak labels stored in the specified JSON file.

- **file_name_data_train**: Contains the name of the .xlsx file storing the training dataset. Used in Step 1 and Step 3.1.

- **sentence_col_name**: Indicates the name of the column in the .xlsx file containing the sentences or texts to be classified. Used in Step 1, Step 2, and Step 3.1.

- **sample_size**: Defines the size of the sample used in the weak labeling process. It can be an integer or a decimal value between 0 and 1, representing the fraction of the total dataset used for generating weak labels.

- **seed**: The seed value used for generating random numbers, ensuring reproducibility when the same seed is used across different script executions.

- **root_labels**: Path to the folder where the weak labels generated through the binary question-based labeling process will be stored. In Step 2, this path also points to the folder where the pre-generated weak labels are stored. This variable specifies the location where the weak labels are stored and will not be used to store new information.

- **temp**: Controls the temperature applied to the GPT-3.5-turbo model while generating responses. A higher value increases randomness, while a lower value produces more deterministic responses.

- **max_t**: The maximum number of tokens allowed in the responses generated by the model. It's used to limit the length of the responses to a specific size.

- **verbose**: A boolean value that controls whether status and progress messages will be printed during script execution.

- **model_name**: Name of the base model to be used for the NLLF generator. By default, "bert-base-uncased" is used.

- **maxlen_s**: Maximum token length for the text to be tokenized. Longer texts will be truncated.

- **maxlen_bsq**: Maximum token length for the Binary SubTask Questions (BSQs).

- **batch_size**: Batch size for training. Specifies how many examples are processed together in each training iteration.

- **epochs**: Number of training epochs. An epoch represents a complete pass through the training dataset.

- **lr**: Learning rate for training. Controls how much the model's weights are adjusted based on the error.

- **hf_token**: Hugging Face User Access Token required to interact with their services, such as loading trained models.

- **repo_name**: Name of the repository where the NLLF generator will be saved. In Step 3.1, this refers to the repository on the Hugging Face platform where the pre-trained NLLF generator is stored.

- **username**: Hugging Face username associated with the repository.

- **file_name_new_dict_bsqs**: Represents the name of the JSON file containing the Binary SubTask Questions (BSQs) necessary for constructing the Natural Language Learned Feature (NLLF) representation.

- **file_name_data_val**: Contains the name of the .xlsx file storing the validation dataset.

- **file_name_data_test**: Contains the name of the .xlsx file storing the test dataset.

- **root_labels_in**: Path to the folder where the NLLF representations generated through the binary question-based labeling process using the pre-trained generator will be stored. The root folder path indicate where the NLLF features are located, which will be used as input for the integration process.

- **file_name_support**: Here, the filename of a .txt file is provided. This file contains a list of NLLF feature names that will be used in the integration process.

- **label_col_name**: This argument sets the name of the column containing the task labels in the data. Task labels are the outputs that the decision tree model will attempt to predict.

- **dt_max_depth**: This argument sets the maximum depth for the Decision Tree (DT). It controls the complexity of the tree and thus can influence its ability to fit the data. The default value is 5 if not explicitly provided.

- **root_labels_out**: This option sets the root folder path where the resulting predictions and model parameters from the feature integration will be saved.


### One-line for non-expert

Execute the following command to utilize the NLLF pipeline:

```
python method/one_line.py \
  --api_key <your_api_key_value> \
  --file_name_dict_bsqs <your_file_name_dict_bsqs_value> \
  --file_name_data_train <your_file_name_data_train_value> \
  --sentence_col_name <your_sentence_col_name_value> \
  --sample_size <your_sample_size_value> \
  --seed <your_seed_value> \
  --root_labels <your_root_labels_value> \
  --temp <your_temperature_value> \
  --max_t <your_max_tokens_value> \
  --verbose <your_verbose_value> \
  --model_name bert-base-uncased \
  --maxlen_s <your_max_token_length_s_value> \
  --maxlen_bsq <your_max_token_length_bsq_value> \
  --batch_size <your_batch_size_value> \
  --epochs <your_epochs_value> \
  --lr <your_learning_rate_value> \
  --hf_token <your_huggingface_token_value> \
  --repo_name <your_repository_name_value> \
  --username <your_huggingface_username_value>
  --file_name_new_dict_bsqs <your_file_name_new_dict_bsqs_value> \
  --file_name_data_val <your_file_name_data_val_value> \
  --file_name_data_test <your_file_name_data_test_value> \ 
  --root_labels_in <your_root_labels_in_value> \
  --file_name_support <your_file_name_support_value> \
  --label_col_name <your_label_col_name_value> \
  --dt_max_depth <your_decision_tree_max_depth_value> \
  --root_labels_out <your_root_labels_out_value>
```

Feel free to tailor the parameters to your specific needs and unlock the capabilities of the NLLF pipeline.

For more detailed explanations and insights into each parameter and step, please refer to the main documentation (Step-by-step). Happy NLLF generating!

### Step-by-step

#### Step 1: Zero-shot Sub-task Labelisation

```
python method/01_one_line.py \
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

#### Step 2: Training of NLLF Generator

```
python method/02_one_line.py \
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

#### Step 3.1: NLLF Generation

```
python method/03_1_one_line.py \
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

#### Step 3.2: NLLF Integration

```
python method/03_2_one_line.py \
  --root_labels_in <your_root_labels_in_value> \
  --file_name_support <your_file_name_support_value> \
  --label_col_name <your_label_col_name_value> \
  --dt_max_depth <your_decision_tree_max_depth_value> \
  --root_labels_out <your_root_labels_out_value>
```

### Experiments




### Citation

If you find this repo useful, please cite our paper:
```
@article{nllf2023,
  title={Deep Natural Language Feature Learning for Interpretable Prediction},
  author={Anonymous EMNLP submission},
  year={2023}
}