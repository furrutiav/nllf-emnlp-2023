from class_nllfg_integration import NLLFIntergration
import argparse

"""_Example_
python 03_2_one_line.py --root_labels_in 02_labels --file_name_support data/support.txt --label_col_name label --dt_max_depth 5 --root_labels_out 03_model_predictions
"""

if __name__ == "__main__":
    """_Parameters description_
    root_labels_in:     This option specifies the root folder path where the NLLF features (files or data) are located, 
                        which will be used as input for the integration process.
    file_name_support:  Here, the filename of a .txt file is provided. This file contains the names of the NLLF features 
                        that will be used in the process. The file could contain a list of relevant feature names.
    label_col_name:     This argument sets the name of the column containing the task labels in the data. Task labels are 
                        the outputs that the decision tree model will attempt to predict.
    dt_max_depth:       This argument sets the maximum depth for the Decision Tree (DT). It controls the complexity of the 
                        tree and thus can influence its ability to fit the data. The default value is 5 if not explicitly 
                        provided.
    root_labels_out:    This option sets the root folder path where the resulting predictions and model parameters from the 
                        feature integration will be saved.
    """
    argParser = argparse.ArgumentParser()
    
    # Instance
    argParser.add_argument("-li", "--root_labels_in", help="Root folder of the NLL (Natural Language Learned) features")
    argParser.add_argument("-fs", "--file_name_support", help="File name of your .txt with support NLLF")
    argParser.add_argument("-c", "--label_col_name", help="Column name of your task-label")
    argParser.add_argument("-md", "--dt_max_depth", default = 5, help="Max. depth for the Decision Tree (DT)")

    # Save
    argParser.add_argument("-lo", "--root_labels_out", help="Root folder for predictions and model parameters")
    
    args = argParser.parse_args()
    
    root_labels_in = args.root_labels_in
    file_name_support = args.file_name_support
    label_col_name = args.label_col_name
    dt_max_depth = int(args.dt_max_depth)
    
    root_labels_out = args.root_labels_out
    
    # Print the parameters
    print("root_labels_in:", root_labels_in)
    print("file_name_support:", file_name_support)
    print("label_col_name:", label_col_name)
    print("dt_max_depth:", dt_max_depth)
    print("root_labels_out:", root_labels_out)

    integrator = NLLFIntergration(
        root_labels = root_labels_in,
        file_name_support = file_name_support,
        label_col_name = label_col_name,
        dt_max_depth=dt_max_depth
    )
    
    integrator.save_predict(
        root_labels = root_labels_out
    )
        
    print("Success!")