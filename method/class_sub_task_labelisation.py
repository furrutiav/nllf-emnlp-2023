import pandas as pd
import openai
import time
import json

    
class SubTaskLabelisator:
    def __init__(self, api_key, file_name_dict_bsqs, file_name_data_train, sentence_col_name, sample_size, seed=2023):
        """
        chatgpt_template = lambda sen, bsq: f"Sentence: {sen}
        Based on the sentence, {bsq} (answer 'Yes' or 'No')"
        """
        openai.organization = ""
        openai.api_key = api_key
        
        self.dict_bsqs = {}
        with open(file_name_dict_bsqs, 'r') as f:
            self.dict_bsqs = json.load(f)
    
        self.data_train = pd.read_excel(file_name_data_train, index_col=0)
        self.sample_size = sample_size
        
        self.sentence_col_name = sentence_col_name
        
        _cased_ = sentence_col_name[0].upper() + sentence_col_name[1:]
        _uncased_ = sentence_col_name[0].lower() + sentence_col_name[1:]
        
        self.chatgpt_template = lambda sen, bsq: f"""{_cased_}: {sen}\nBased on the {_uncased_}, {bsq} (answer 'Yes' or 'No')"""
        
        self.seed = seed

        self.sample_train = self.get_sample(sample_size, seed)

    def get_sample(self, sample_size, seed):
        if sample_size <= 1:
            return self.data_train.sample(frac = sample_size, random_state=seed)
        else:
            return self.data_train.sample(sample_size, random_state=seed)

    def chatgpt(self, text, temp, max_t):
        try: 
            output = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": text},
                ],
                temperature=temp,
                max_tokens=max_t,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0)
        except:
            print("Thinking...")
            time.sleep(1)
            return self.chatgpt(text, temp=temp, max_t=max_t)
    
        return output["choices"][0]["message"]["content"]

    def run_labeling(self, root_labels, temp=0, max_t=10, verbose=False):
        for key, b in list(self.dict_bsqs.items()):
            if verbose: print(key, "BSQ:", b)
            responses = []
            for i, ix in enumerate(self.sample_train.index):
                s = self.sample_train.loc[ix][self.sentence_col_name]
                o = self.chatgpt(self.chatgpt_template(s, b), temp, max_t)
                responses.append([ix, o])
                if verbose and ((i+1) % 10 == 0): print(f"{i+1}/{self.sample_train.shape[0]}")
            train_bi = self.sample_train.copy()
            train_bi["chatgpt"] = [x[1] for x in responses]

            train_bi.to_excel(f"{root_labels}/chatgpt_responses_{key}.xlsx")
        pass
