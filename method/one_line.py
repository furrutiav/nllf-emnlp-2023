from class_sub_task_labelisation import SubTaskLabelisator
from class_nllfg_training import NLLFGeneratorTraining
from class_nllf_generation import NLLFGeneratorInAction
from class_nllfg_integration import NLLFIntergration
import argparse


"""_Example_
python one_line.py --api_key OPENAI-API-KEY --file_name_dict_bsqs data/dict_bsqs.json --file_name_data_train data/data_train.xlsx --sentence_col_name abstract --sample_size 100 --seed 42 --root_labels 01_labels --temp 0 --max_t 5 --verbose True --model_name bert-base-uncased --maxlen_s 489 --maxlen_bsq 20 --batch_size 8 --epochs 5 --lr 2e-5 --hf_token HF-TOKEN --repo_name example_juke --username HF-USERNAME --file_name_new_dict_bsqs data/new_dict_bsqs.json --file_name_data_val data/data_val.xlsx --file_name_data_test data/data_test.xlsx --root_labels_in 02_label --file_name_support data/support.txt --label_col_name label --dt_max_depth 5 --root_labels_out 03_model_predictions
"""

if __name__ == "__main__":
    """_Parameters description_
    api_key:                    (Step 1) Stores the OpenAI API key, necessary for authenticating and accessing the GPT-3.5-turbo 
                                language model. This key enables communication with OpenAI's services to perform weak labeling 
                                tasks.
    file_name_dict_bsqs:        (Step 1) Represents the name of the JSON file containing the Binary SubTask Questions (BSQ) used 
                                in the weak labeling strategy. These questions are formulated to generate weak labels for sentences 
                                in the dataset. (Step 2) At the same time, the file is used as references to load pre-generated weak 
                                labels. These questions will serve as guides to load the weak labels stored in the specified JSON file.
    file_name_data_train:       (Step 1, 3.1) Contains the name of the .xlsx file storing the training dataset. 
    sentence_col_name:          (Step 1, 2, 3.1) Indicates the name of the column in the .xlsx file containing the sentences or texts-
                                to-classify.
    sample_size:                (Step 1) This variable defines the size of the sample used in the weak labeling process. It can be an 
                                integer or a decimal value between 0 and 1, representing the fraction of the total dataset used 
                                for generating weak labels.
    seed:                       (Step 1) The seed value used for generating random numbers, ensuring reproducibility when the same seed 
                                is used across different script executions.
    root_labels:                (Step 1) Path to the folder where the weak labels generated through the binary question-based 
                                labeling process will be stored. (Step 2) Path to the folder where the pre-generated weak labels 
                                are stored. This variable specifies the location where the weak labels are stored and will not be 
                                used to store new information.
    temp:                       (Step 1) Controls the temperature applied to the GPT-3.5-turbo model while generating responses. A 
                                higher value increases randomness, while a lower value produces more deterministic responses.
    max_t:                      (Step 1) The maximum number of tokens allowed in the responses generated by the model. It's used to 
                                limit the length of the responses to a specific size.
    verbose:                    (Step 1, 2, 3.1) A boolean value that controls whether status and progress messages will be printed 
                                during script execution.
    model_name:                 (Step 2) Name of the base model to be used for the NLLF generator. By default, "bert-base-uncased" 
                                is used.
    maxlen_s:                   (Step 2, 3.1) Maximum token length for the text to be tokenized. Longer texts will be truncated.
    maxlen_bsq:                 (Step 2, 3.1) Maximum token length for the Binary SubTask Questions (BSQs).
    batch_size:                 (Step 2) Batch size for training. Specifies how many examples are processed together in each training
                                iteration.
    epochs:                     (Step 2) Number of training epochs. An epoch represents a complete pass through the training dataset.
    lr:                         (Step 2) Learning rate for training. Controls how much the model's weights are adjusted based on the 
                                error.
    hf_token:                   (Step 2) Hugging Face User Access Token required to interact with their services, such as loading 
                                trained models.
    repo_name:                  (Step 2) Name of the repository where the NLLF generator will be saved. (Step 3.1) The name of the 
                                repository on the Hugging Face platform where the pre-trained NLLF generator is stored.
    username:                   (Step 2, 3.1) Hugging Face username associated with the repository.
    file_name_new_dict_bsqs:    Represents the name of the JSON file containing the Binary SubTask Questions (BSQs) 
                                necessary for constructing the Natural Language Learned Feature (NLLF) representation.
    file_name_data_val:         (Step 3.1) Contains the name of the .xlsx file storing the validation dataset.
    file_name_data_test:        (Step 3.1) Contains the name of the .xlsx file storing the test dataset.
    root_labels_in:             (Step 3.1) Path to the folder where the NLLF representations generated through the binary question-based 
                                labeling process using the pre-trained generator will be stored. (Step 3.2) This option specifies the root 
                                folder path where the NLLF features (files or data) are located, which will be used as input for the 
                                integration process.
    file_name_support:          (Step 3.2) Here, the filename of a .txt file is provided. This file contains the names of the NLLF features 
                                that will be used in the process. The file contain a list of NLLF feature names.
    label_col_name:             (Step 3.2) This argument sets the name of the column containing the task labels in the data. Task labels are 
                                the outputs that the decision tree model will attempt to predict.
    dt_max_depth:               (Step 3.2) This argument sets the maximum depth for the Decision Tree (DT). It controls the complexity of the 
                                tree and thus can influence its ability to fit the data. The default value is 5 if not explicitly 
                                provided.  
    root_labels_out:            (Step 3.2) This option sets the root folder path where the resulting predictions and model parameters from the 
                                feature integration will be saved.                          
    """
    argParser = argparse.ArgumentParser()
    
    # Step 1
    ## Instance
    argParser.add_argument("-ak", "--api_key", help="OpenAI - API Key")
    argParser.add_argument("-b", "--file_name_dict_bsqs", help="File name of your JSON with BSQs")
    argParser.add_argument("-d", "--file_name_data_train", help="File name of your .xlsx Training Dataset")
    argParser.add_argument("-c", "--sentence_col_name", help="Column name of your text-to-classify")
    argParser.add_argument("-s", "--sample_size", default = 0.1, help="Sample size for zero-shot labelisation: Integer number or fraction between 0 to 1")
    argParser.add_argument("-r", "--seed", default = 2023, help="Random seed")
    
    ## Run
    argParser.add_argument("-l", "--root_labels", help="Root folder to save the weak-labels")
    argParser.add_argument("-t", "--temp", default = 0, help="Temperature of GPT-3.5-turbo")
    argParser.add_argument("-m", "--max_t", default = 5, help="Max. number of output-tokens")
    argParser.add_argument("-v", "--verbose", default = False, help="Print status")
    
    # Step 2
    ## Instance
    argParser.add_argument("-mn", "--model_name", default="bert-base-uncased", help="Base model name for your NLLF generator (This version: Only for BERT models from HuggingFace)") 
    argParser.add_argument("-ms", "--maxlen_s", default = 489, help="Max. number of tokens for your tokenize text-to-classify")
    argParser.add_argument("-mb", "--maxlen_bsq", default = 20, help="Max. number of tokens for your tokenize BSQs")
    argParser.add_argument("-bs", "--batch_size", default = 8, help="Batch size for the training")
    ## Train
    argParser.add_argument("-e", "--epochs", default = 5, help="Number of epochs for the training")
    argParser.add_argument("-lr", "--lr", default = 2e-5, help="Learning rate for the training")
    ## Save
    argParser.add_argument("-hft", "--hf_token", help="Hugging Face User Access Token")
    argParser.add_argument("-rn", "--repo_name", help="Repo. name for your NLLF generator")
    argParser.add_argument("-u", "--username", help="Hugging Face Username")
    
    # Step 3.1
    ## Instance
    argParser.add_argument("-nb", "--file_name_new_dict_bsqs", help="File name of your JSON with new BSQs")
    argParser.add_argument("-dv", "--file_name_data_val", help="File name of your .xlsx Validation Dataset")
    argParser.add_argument("-dte", "--file_name_data_test", help="File name of your .xlsx Testing Dataset")
    
    # Step 3.2
    ## Instance
    argParser.add_argument("-li", "--root_labels_in", help="Root folder of the NLL (Natural Language Learned) features")
    argParser.add_argument("-fs", "--file_name_support", help="File name of your .txt with support NLLF")
    argParser.add_argument("-lc", "--label_col_name", help="Column name of your task-label")
    argParser.add_argument("-md", "--dt_max_depth", default = 5, help="Max. depth for the Decision Tree (DT)")
    # Save
    argParser.add_argument("-lo", "--root_labels_out", help="Root folder for predictions and model parameters")
    
    args = argParser.parse_args()
    
    api_key = args.api_key
    file_name_dict_bsqs = args.file_name_dict_bsqs
    file_name_data_train = args.file_name_data_train
    sentence_col_name = args.sentence_col_name
    sample_size = float(args.sample_size) if "0." in args.sample_size else int(args.sample_size)
    seed = int(args.seed)
    
    root_labels = args.root_labels
    temp = float(args.temp)
    max_t = int(args.max_t)
    verbose = bool(args.verbose)
    
    model_name = args.model_name
    maxlen_s = int(args.maxlen_s)
    maxlen_bsq = int(args.maxlen_bsq)
    batch_size = int(args.batch_size)
    
    epochs = int(args.epochs)
    lr = float(args.lr)
    
    hf_token = args.hf_token
    repo_name = args.repo_name
    username = args.username
    
    file_name_new_dict_bsqs = args.file_name_new_dict_bsqs
    file_name_data_val = args.file_name_data_val
    file_name_data_test = args.file_name_data_test
    
    root_labels_in = args.root_labels_in
    file_name_support = args.file_name_support
    label_col_name = args.label_col_name
    dt_max_depth = int(args.dt_max_depth)
    
    root_labels_out = args.root_labels_out
    
    # Print the parameters
    print("api_key:", api_key)
    print("file_name_dict_bsqs:", file_name_dict_bsqs)
    print("file_name_data_train:", file_name_data_train)
    print("sentence_col_name:", sentence_col_name)
    print("sample_size:", sample_size)
    print("seed:", seed)
    print("root_labels:", root_labels)
    print("temp:", temp)
    print("max_t:", max_t)
    print("verbose:", verbose)
    print("model_name:", model_name)
    print("maxlen_s:", maxlen_s)
    print("maxlen_bsq:", maxlen_bsq)
    print("batch_size:", batch_size)
    print("epochs:", epochs)
    print("lr:", lr)
    print("hf_token:", hf_token)
    print("repo_name:", repo_name)
    print("username:", username)
    print("file_name_new_dict_bsqs:", file_name_new_dict_bsqs)
    print("file_name_data_val:", file_name_data_val)
    print("file_name_data_test:", file_name_data_test)
    print("root_labels_in:", root_labels_in)
    print("file_name_support:", file_name_support)
    print("label_col_name:", label_col_name)
    print("dt_max_depth:", dt_max_depth)
    print("root_labels_out:", root_labels_out)
    
    # Step 1

    labelisator = SubTaskLabelisator(
        api_key = api_key,
        file_name_dict_bsqs = file_name_dict_bsqs, 
        file_name_data_train = file_name_data_train,
        sentence_col_name = sentence_col_name,
        sample_size = sample_size,
        seed = seed
    )
    
    labelisator.run_labeling(
        root_labels = root_labels, 
        temp=temp,      
        max_t=max_t,    
        verbose=verbose 
    )
    
    print("Step 1: Success!")
    
    # Step 2
    
    training = NLLFGeneratorTraining(
        file_name_dict_bsqs = file_name_dict_bsqs,
        root_labels = root_labels,
        sentence_col_name = sentence_col_name,
        model_name = model_name,
        maxlen_s=maxlen_s,
        maxlen_bsq=maxlen_bsq,
        batch_size=batch_size
    )
    
    training.train(
        epochs=epochs,
        lr=lr,
        verbose=verbose
    )
    
    training.save(
        hf_token = hf_token,
        repo_name = repo_name,
        username= username
    )
        
    print("Step 2: Success!")
    
    # Step 3.1
    
    generator = NLLFGeneratorInAction(
        file_name_new_dict_bsqs = file_name_new_dict_bsqs,
        maxlen_s = maxlen_s,
        maxlen_bsq = maxlen_bsq,
        username = username,
        repo_name = repo_name,
        file_name_data_train = file_name_data_train,
        file_name_data_val = file_name_data_val,  
        file_name_data_test = file_name_data_test,
        sentence_col_name = sentence_col_name
    )
    
    generator.apply(
        root_labels=root_labels_in,
        verbose=verbose
    )
        
    print("Step 3.1: Success!")
    
    # Step 3.2
    
    integrator = NLLFIntergration(
        root_labels = root_labels_in,
        file_name_support = file_name_support,
        label_col_name = label_col_name,
        dt_max_depth=dt_max_depth
    )
    
    integrator.save_predict(
        root_labels = root_labels_out
    )
    
    print("Step 3.2: Success!")
    
    print("Success!")
