import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_url, cached_download
from transformers import BertTokenizer, BertModel
import torch
import os
import json

print("torch.cuda.is_available()")
assert(torch.cuda.is_available() == True)
print(torch.cuda.is_available())

class NLLFGeneratorInAction:

    def __init__(self, file_name_new_dict_bsqs, maxlen_s, maxlen_bsq, username, repo_name, file_name_data_train, file_name_data_val, file_name_data_test, sentence_col_name):
        self.new_dict_bsqs = {}
        with open(file_name_new_dict_bsqs, 'r') as f:
            self.new_dict_bsqs = json.load(f)
            
        self.sentence_col_name = sentence_col_name
        
        self.maxlen_s = maxlen_s
        self.maxlen_bsq = maxlen_bsq
        
        config_file_url = hf_hub_url(f"{username}/{repo_name}", filename="cls_layer.torch")
        value = cached_download(config_file_url)
        self.cls_layer = torch.load(value)
        
        self.the_model = BertModel.from_pretrained(f"{username}/{repo_name}").cuda()
        self.the_tokenizer = BertTokenizer.from_pretrained(f"{username}/{repo_name}", do_lower_case=False)

        self.data_train = pd.read_excel(file_name_data_train, index_col=0)
        self.data_val = pd.read_excel(file_name_data_val, index_col=0)
        self.data_test = pd.read_excel(file_name_data_test, index_col=0)


    def _preproccesing(self, sen, b):
        sentence1 = str(sen)
        sentence2 = str(b)
            
        tokens1 = self.the_tokenizer.tokenize(sentence1) if len(sentence1)>0 else ["[UNK]"]
        tokens2 = self.the_tokenizer.tokenize(sentence2) if len(sentence2)>0 else ["[UNK]"]
        
        if len(tokens1) <= self.maxlen_s:
            tokens1 = tokens1 + ['[PAD]' for _ in range(self.maxlen_s - len(tokens1))]
        else:
            tokens1 = tokens1[:self.maxlen_s]

        if len(tokens2) <= self.maxlen_bsq:
            tokens2 = tokens2 + ['[PAD]' for _ in range(self.maxlen_bsq - len(tokens2))]
        else:
            tokens2 = tokens2[:self.maxlen_bsq]
            
        tokens = ["[CLS]"]+tokens1+["[SEP]"]+tokens2+["[SEP]"]
        tokens_ids = self.the_tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)
        attn_mask = (tokens_ids_tensor != 0).long() # [PAD] => 1

        return tokens_ids_tensor.cuda(), attn_mask.cuda()

    def decisionClassifier(self, sen, b):
        tokens_ids_tensor, attn_mask = self._preproccesing(sen, b)
        cont_reps = self.the_model(tokens_ids_tensor.unsqueeze(0), attention_mask = attn_mask.unsqueeze(0))
        cls_rep = cont_reps.last_hidden_state[:, 0]
        logits = self.cls_layer(cls_rep)
        probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy()[0]

    
    def _predict(self, root_labels, verbose):
        for i, kb in enumerate(self.new_dict_bsqs.keys()):

            bsq = self.new_dict_bsqs[kb]

            if verbose: print("[train] ChatGPT question:", kb)
            new_train = self.data_train.copy()
            y_pred = new_train.apply(lambda x: self.decisionClassifier(str(x[self.sentence_col_name]), bsq), axis=1)
            new_train["juke"] = y_pred
            new_train.to_excel(f"{root_labels}/train_nllf_{kb}.xlsx")
            
            if verbose: print("[test] ChatGPT question:", kb)
            new_test = self.data_test.copy()
            y_pred = new_test.apply(lambda x: self.decisionClassifier(str(x[self.sentence_col_name]), bsq), axis=1)
            new_test["juke"] = y_pred
            new_test.to_excel(f"{root_labels}/test_nllf_{kb}.xlsx")
            
            if verbose: print("[dev] ChatGPT question:", kb)
            new_dev = self.data_val.copy()
            y_pred = new_dev.apply(lambda x: self.decisionClassifier(str(x[self.sentence_col_name]), bsq), axis=1)
            new_dev["juke"] = y_pred
            new_dev.to_excel(f"{root_labels}/val_nllf_{kb}.xlsx")

        pass

    def _ensemble(self, root_labels):
        
        file_names = [] 
        for file_name in os.listdir(f"{root_labels}/"):
            if (".xlsx" in file_name) and (file_name not in ["train_nllf.xlsx", "val_nllf.xlsx", "test_nllf.xlsx"]):
                file_names.append(file_name)
        file_names

        file_names_per_set = {
            "train": [],
            "val": [],
            "test": []
        }
        for x in file_names:
            file_names_per_set[x.split("_")[0]].append(f"{root_labels}/{x}")

        nllf_per_set = {}
        for k in file_names_per_set.keys():
            u = file_names_per_set[k][0]
            o = pd.read_excel(u, index_col=0)
            l = "_".join(u.split("/")[1].replace(".xlsx", "").split("_")[2:])
            o[f"{l}(N)"] = o["juke"].apply(lambda x: float(list(x[1:-1].split())[0]))
            o[f"{l}(Y)"] = o["juke"].apply(lambda x: float(list(x[1:-1].split())[1]))
            o = o.drop(columns="juke")
            nllf_per_set[k] = o.copy()
            for u in file_names_per_set[k][1:]:
                print(u)
                o = pd.read_excel(u, index_col=0)
                l = "_".join(u.split("/")[1].replace(".xlsx", "").split("_")[2:])
                nllf_per_set[k][f"{l}(N)"] = o["juke"].apply(lambda x: float(list(x[1:-1].split())[0]))
                nllf_per_set[k][f"{l}(Y)"] = o["juke"].apply(lambda x: float(list(x[1:-1].split())[1]))

        for k in file_names_per_set.keys():
            nllf_per_set[k].to_excel(f"{root_labels}/{k}_nllf.xlsx")
        
        pass

    def apply(self, root_labels, verbose=False):

        self._predict(root_labels, verbose)

        self._ensemble(root_labels)

        pass