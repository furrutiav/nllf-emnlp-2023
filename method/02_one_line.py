from class_nllfg_training import NLLFGeneratorTraining
import argparse

"""_Example_
python 02_one_line.py --file_name_dict_bsqs data/dict_bsqs.json --root_labels 01_labels --sentence_col_name abstract --model_name bert-base-uncased --maxlen_s 489 --maxlen_bsq 20 --batch_size 8 --epochs 5 --lr 2e-5 --verbose False --hf_token HF-TOKEN --repo_name example_juke --username HF-USERNAME
"""

if __name__ == "__main__":
    """_Parameters description_
    file_name_dict_bsqs:    Name of the JSON file containing Binary SubTask Questions (BSQs) used as references to load 
                            pre-generated weak labels. These questions will serve as guides to load the weak labels 
                            stored in the specified JSON file.
    root_labels:            Path to the folder where the pre-generated weak labels are stored. This variable specifies 
                            the location where the weak labels are stored and will not be used to store new information.
    sentence_col_name:      Indicates the name of the column in the .xlsx file containing the sentences or texts that 
                            will undergo the weak labeling process.
    model_name:             Name of the base model to be used for the NLLF generator. By default, "bert-base-uncased" 
                            is used.
    maxlen_s:               Maximum token length for the text to be tokenized. Longer texts will be truncated.
    maxlen_bsq:             Maximum token length for the Binary SubTask Questions (BSQs).
    batch_size:             Batch size for training. Specifies how many examples are processed together in each training
                            iteration.
    epochs:                 Number of training epochs. An epoch represents a complete pass through the training dataset.
    lr:                     Learning rate for training. Controls how much the model's weights are adjusted based on the 
                            error.
    verbose:                A boolean value that determines whether status and progress messages will be printed during 
                            script execution, providing information about the weak labeling process.
    hf_token:               Hugging Face User Access Token required to interact with their services, such as loading 
                            trained models.
    repo_name:              Name of the repository where the NLLF generator will be saved.
    username:               Hugging Face username associated with the repository.
    seed:                   Seed value used for generating random numbers to ensure reproducibility across different 
                            script executions.
    """
    argParser = argparse.ArgumentParser()
    
    # Instance
    argParser.add_argument("-b", "--file_name_dict_bsqs", help="File name of your JSON with BSQs")
    argParser.add_argument("-l", "--root_labels", help="Root folder of the weak-labels")
    argParser.add_argument("-c", "--sentence_col_name", help="Column name of your text-to-classify")
    argParser.add_argument("-mn", "--model_name", default="bert-base-uncased", help="Base model name for your NLLF generator (This version: Only for BERT models from HuggingFace)")  
    argParser.add_argument("-ms", "--maxlen_s", default = 489, help="Max. number of tokens for your tokenize text-to-classify")
    argParser.add_argument("-mb", "--maxlen_bsq", default = 20, help="Max. number of tokens for your tokenize BSQs")
    argParser.add_argument("-bs", "--batch_size", default = 8, help="Batch size for the training")
    
    # Train
    argParser.add_argument("-e", "--epochs", default = 5, help="Number of epochs for the training")
    argParser.add_argument("-lr", "--lr", default = 2e-5, help="Learning rate for the training")
    argParser.add_argument("-v", "--verbose", default = False, help="Print status")
    
    # Save
    argParser.add_argument("-t", "--hf_token", help="Hugging Face User Access Token")
    argParser.add_argument("-rn", "--repo_name", help="Repo. name for your NLLF generator")
    argParser.add_argument("-u", "--username", help="Hugging Face Username")
    
    args = argParser.parse_args()
    
    file_name_dict_bsqs = args.file_name_dict_bsqs
    root_labels = args.root_labels
    sentence_col_name = args.sentence_col_name
    model_name = args.model_name
    maxlen_s = int(args.maxlen_s)
    maxlen_bsq = int(args.maxlen_bsq)
    batch_size = int(args.batch_size)
    
    epochs = int(args.epochs)
    lr = float(args.lr)
    verbose = bool(args.verbose)
    
    hf_token = args.hf_token
    repo_name = args.repo_name
    username = args.username
    
    # Print the parameters
    print("file_name_dict_bsqs:", file_name_dict_bsqs)
    print("root_labels:", root_labels)
    print("sentence_col_name:", sentence_col_name)
    print("model_name:", model_name)
    print("maxlen_s:", maxlen_s)
    print("maxlen_bsq:", maxlen_bsq)
    print("batch_size:", batch_size)
    print("epochs:", epochs)
    print("lr:", lr)
    print("verbose:", verbose)
    print("hf_token:", hf_token)
    print("repo_name:", repo_name)
    print("username:", username)

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
        
    print("Success!")