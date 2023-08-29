from class_sub_task_labelisation import SubTaskLabelisator
import argparse

"""_Example_
python 01_one_line.py --api_key OPENAI-API-KEY --file_name_dict_bsqs data/dict_bsqs.json --file_name_data_train data/data_train.xlsx --sentence_col_name abstract --sample_size 100 --seed 42 --root_labels 01_labels --temp 0 --max_t 5 --verbose True
"""

if __name__ == "__main__":
    """_Parameters description_
    api_key:                Stores the OpenAI API key, necessary for authenticating and accessing the GPT-3.5-turbo 
                            language model. This key enables communication with OpenAI's services to perform weak labeling 
                            tasks.
    file_name_dict_bsqs:    Represents the name of the JSON file containing the Binary SubTask Questions (BSQ) used in the 
                            weak labeling strategy. These questions are formulated to generate weak labels for sentences in 
                            the dataset.
    file_name_data_train:   Contains the name of the .xlsx file storing the training dataset. This dataset serves as input 
                            to generate weak labels using the GPT-3.5-turbo model and the binary questions.
    sentence_col_name:      Indicates the name of the column in the .xlsx file containing the sentences or texts that will 
                            undergo the weak labeling process.
    sample_size:            This variable defines the size of the sample used in the weak labeling process. It can be an 
                            integer or a decimal value between 0 and 1, representing the fraction of the total dataset used 
                            for generating weak labels.
    seed:                   The seed value used for generating random numbers, ensuring reproducibility when the same seed 
                            is used across different script executions.
    root_labels:            Path to the folder where the weak labels generated through the binary question-based labeling 
                            process will be stored.
    temp:                   Controls the temperature applied to the GPT-3.5-turbo model while generating responses. A 
                            higher value increases randomness, while a lower value produces more deterministic responses.
    max_t:                  The maximum number of tokens allowed in the responses generated by the model. It's used to 
                            limit the length of the responses to a specific size.
    verbose:                A boolean value that controls whether status and progress messages will be printed during 
                            script execution.    
"""
    argParser = argparse.ArgumentParser()
    
    # Instance
    argParser.add_argument("-ak", "--api_key", help="OpenAI - API Key")
    argParser.add_argument("-b", "--file_name_dict_bsqs", help="File name of your JSON with BSQs")
    argParser.add_argument("-d", "--file_name_data_train", help="File name of your .xlsx Training Dataset")
    argParser.add_argument("-c", "--sentence_col_name", help="Column name of your text-to-classify")
    argParser.add_argument("-s", "--sample_size", default = 0.1, help="Sample size for zero-shot labelisation: Integer number or fraction between 0 to 1")
    argParser.add_argument("-r", "--seed", default = 2023, help="Random seed")
    
    # Run
    argParser.add_argument("-l", "--root_labels", help="Root folder to save the weak-labels")
    argParser.add_argument("-t", "--temp", default = 0, help="Temperature of GPT-3.5-turbo")
    argParser.add_argument("-m", "--max_t", default = 5, help="Max. number of output-tokens")
    argParser.add_argument("-v", "--verbose", default = False, help="Print status")
    
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
    
    print("Success!")