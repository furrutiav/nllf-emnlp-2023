from class_nllf_generation import NLLFGeneratorInAction
import argparse

"""_Example_
python 03_1_one_line.py --file_name_new_dict_bsqs data/new_dict_bsqs.json --maxlen_s 489 --maxlen_bsq 20 --username HF-USERNAME --repo_name example_juke --file_name_data_train data/data_train.xlsx --file_name_data_val data/data_val.xlsx --file_name_data_test data/data_test.xlsx --sentence_col_name abstract --root_labels 02_labels --verbose False
"""

if __name__ == "__main__":
    """_Parameters description_
    file_name_new_dict_bsqs:    Represents the name of the JSON file containing the Binary SubTask Questions (BSQs) 
                                necessary for constructing the Natural Language Learned Feature (NLLF) representation.
    maxlen_s:                   The maximum number of tokens allowed in the text that will be used for the classification
                                process.
    maxlen_bsq:                 The maximum number of tokens allowed in the context sequences of block selection questions 
                                (BSQs).
    username:                   The Hugging Face username, required for accessing the platform and functionalities related 
                                to natural language processing.
    repo_name:                  The name of the repository on the Hugging Face platform where the pre-trained NLLF generator 
                                is stored.
    file_name_data_train:       Contains the name of the .xlsx file storing the training dataset.
    file_name_data_val:         Contains the name of the .xlsx file storing the validation dataset.
    file_name_data_test:        Contains the name of the .xlsx file storing the test dataset.
    sentence_col_name:          Indicates the column name in the .xlsx file that contains the sentences or texts that will 
                                undergo the NLLF representation process.
    root_labels:                Path to the folder where the NLLF representations generated through the binary question-based 
                                labeling process using the pre-trained generator will be stored.
    verbose:                    A boolean value that controls whether status and progress messages will be printed during 
                                script execution.
"""
    argParser = argparse.ArgumentParser()
    
    # Instance
    argParser.add_argument("-b", "--file_name_new_dict_bsqs", help="File name of your JSON with new BSQs")
    argParser.add_argument("-ms", "--maxlen_s", default = 489, help="Max. number of tokens for your tokenize text-to-classify")
    argParser.add_argument("-mb", "--maxlen_bsq", default = 20, help="Max. number of tokens for your tokenize BSQs")
    argParser.add_argument("-u", "--username", help="Hugging Face Username")
    argParser.add_argument("-rn", "--repo_name", help="Repo. name for your NLLF generator")
    argParser.add_argument("-dtr", "--file_name_data_train", help="File name of your .xlsx Training Dataset")
    argParser.add_argument("-dv", "--file_name_data_val", help="File name of your .xlsx Validation Dataset")
    argParser.add_argument("-dte", "--file_name_data_test", help="File name of your .xlsx Testing Dataset")
    argParser.add_argument("-c", "--sentence_col_name", help="Column name of your text-to-classify")
    
    # Apply
    argParser.add_argument("-l", "--root_labels", help="Root folder to save the NLL (Natural Language Learned) features")
    argParser.add_argument("-v", "--verbose", default = False, help="Print status")
    
    args = argParser.parse_args()
    
    file_name_new_dict_bsqs = args.file_name_new_dict_bsqs
    maxlen_s = int(args.maxlen_s)
    maxlen_bsq = int(args.maxlen_bsq)
    username = args.username
    repo_name = args.repo_name
    sentence_col_name = args.sentence_col_name
    file_name_data_train = args.file_name_data_train
    file_name_data_val = args.file_name_data_val
    file_name_data_test = args.file_name_data_test
    
    root_labels = args.root_labels
    verbose = bool(args.verbose)
    
    # Print the parameters
    print("file_name_new_dict_bsqs:", file_name_new_dict_bsqs)
    print("maxlen_s:", maxlen_s)
    print("maxlen_bsq:", maxlen_bsq)
    print("username:", username)
    print("repo_name:", repo_name)
    print("sentence_col_name:", sentence_col_name)
    print("file_name_data_train:", file_name_data_train)
    print("file_name_data_val:", file_name_data_val)
    print("file_name_data_test:", file_name_data_test)
    print("root_labels:", root_labels)
    print("verbose:", verbose)

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
        root_labels=root_labels,
        verbose=verbose
    )
        
    print("Success!")