import pandas as pd
import openai
import time

class SubTaskLabelisator:
    def __init__(self, dict_bsqs, data_train, sentence_col_name, sample_size, chatgpt_template, seed=2023):
        """
        chatgpt_template = lambda sen, bsq: f"Sentence: {sen}
        Based on the sentence, {bsq} (answer 'Yes' or 'No')"
        """
        self.dict_bsqs = dict_bsqs
        self.data_train = data_train
        self.sample_size = sample_size
        self.chatgpt_template = chatgpt_template
        self.sentence_col_name = sentence_col_name
        self.seed = seed

        self.sample_train = self.get_sample(sample_size, seed)

    def get_sample(sample_size, seed):
        return self.data_train.sample(sample_size, random_state=seed)

    def chatgpt(text, temp, max_t):
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

    def run_labeling(root_labels, temp=0, max_t=10):
        for key, b in list(self.dict_bsqs.items()):
            responses = []
            for i, ix in enumerate(self.sample_train.index):
                s = self.sample_train.loc[ix][self.sentence_col_name]
                o = self.chatgpt(self.chatgpt_template(s, b), temp, max_t)
                responses.append([ix, o])

            train_bi = self.sample_train.copy()
            train_bi["chatgpt"] = [x[1] for x in responses]

            train_bi.to_excel(f"{root_labels}/chatgpt_responses_{key}.xlsx")
        pass
