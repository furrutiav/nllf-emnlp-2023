import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from huggingface_hub import login as hf_login
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from huggingface_hub import HfApi

print("torch.cuda.is_available()")
assert(torch.cuda.is_available() == True)
print(torch.cuda.is_available())

class DecisionClassifier(nn.Module):
    def __init__(self, model_name):
        super(DecisionClassifier, self).__init__()
        torch.manual_seed(2023)
        
        self.bert_layer = BertModel.from_pretrained(model_name).cuda()
        self.cls_layer = nn.Linear(768, 2).cuda()

    def forward(self, seq, attn_masks):

        cont_reps = self.bert_layer(seq, attention_mask=attn_masks)
        
        cls_rep = cont_reps.last_hidden_state[:, 0]
        
        logits = self.cls_layer(cls_rep)

        return logits

class DatasetTaskDecision(Dataset):
    def __init__(self, df, model_name, maxlen_s, maxlen_bsq, sentence_col_name, bsq_col_name, label_col_name):
        self.df = df
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.maxlen_s = maxlen_s
        self.maxlen_bsq = maxlen_bsq
        self.sentence_col_name = sentence_col_name
        self.bsq_col_name = bsq_col_name
        self.label_col_name = label_col_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence1 = str(self.df.loc[index, self.sentence_col_name])
        sentence2 = str(self.df.loc[index, self.bsq_col_name])

        label = int(self.df.loc[index, self.label_col_name])
        
        tokens1 = self.tokenizer.tokenize(sentence1) if len(sentence1)>0 else ["[UNK]"]
        tokens2 = self.tokenizer.tokenize(sentence2) if len(sentence2)>0 else ["[UNK]"]

        if len(tokens1) <= self.maxlen_s:
            tokens1 = tokens1 + ['[PAD]' for _ in range(self.maxlen_s - len(tokens1))]
        else:
            tokens1 = tokens1[:self.maxlen_s]
            
        if len(tokens2) <= self.maxlen_bsq:
            tokens2 = tokens2 + ['[PAD]' for _ in range(self.maxlen_bsq - len(tokens2))]
        else:
            tokens2 = tokens2[:self.maxlen_bsq]
    
        tokens = ["[CLS]"]+tokens1+["[SEP]"]+tokens2+["[SEP]"]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)
        attn_mask = (tokens_ids_tensor != 0).long() # [PAD] => 0

        return tokens_ids_tensor, attn_mask, label

class NLLFGeneratorTraining:

    def __init__(self, dict_bsqs, root_labels, sentence_col_name, model_name, maxlen_s, maxlen_bsq, batch_size=32):
        self.dict_bsqs = dict_bsqs
        self.sentence_col_name = sentence_col_name
        self.model_name = model_name
        
        self.dataset = self.get_dataset(root_labels)

        self.prepare_split(maxlen_s, maxlen_bsq, batch_size=batch_size)

    def get_dataset(self, root_labels):
        self.label_bsq = "label_bsq"
        self.bsq_col_name = "bsq"

        labels = {}
        for key in self.dict_bsqs.keys():
            labels[key] = pd.read_excel(f"{root_labels}/chatgpt_responses_{key}.xlsx", index_col=0)

        dataset = labels[list(self.dict_bsqs.keys())[0]].copy()
        dataset[self.bsq_col_name] = self.dict_bsqs[list(self.dict_bsqs.keys())[0]]
        for kwy in list(self.dict_bsqs.keys())[1:]:
            o = labels[kwy].copy()
            o[self.bsq_col_name] = self.dict_bsqs[kwy]
            dataset = pd.concat([dataset, o], axis=0, ignore_index=1)
        dataset[self.label_bsq] = dataset["chatgpt"].apply(lambda x: int("yes" in x.lower()))
        dataset = dataset[[self.sentence_col_name, self.bsq_col_name, self.label_bsq]]
        
        return dataset

    def prepare_split(self, maxlen_s=50, maxlen_bsq=20, batch_size=32, num_workers=2, train_ratio=0.9, seed=2023):

        df_train = self.dataset.sample(frac=train_ratio, random_state=seed)
        df_val = self.dataset.loc[[ix for ix in self.dataset.index if ix not in df_train.index]]

        df_train = df_train.reset_index()
        df_val = df_val.reset_index()

        train_set = DatasetTaskDecision(df_train, self.model_name, maxlen_s, maxlen_bsq, 
                                        self.sentence_col_name, self.bsq_col_name, self.label_bsq)
        val_set = DatasetTaskDecision(df_val, self.model_name, maxlen_s, maxlen_bsq,
                                        self.sentence_col_name, self.bsq_col_name, self.label_bsq)

        self.train_loader = DataLoader(
            train_set, 
            batch_size = batch_size, 
            num_workers = num_workers, 
            shuffle=False
            )
        
        self.val_loader = DataLoader(
            val_set, 
            batch_size = batch_size, 
            num_workers = num_workers,
            shuffle=False
        )
        
        pass

    def _train(self, net, criterion, opti, train_loader, val_loader, epochs, verbose=False):
        for ep in range(epochs):
            for it, (seq, attn_masks, labels) in enumerate(train_loader):
                opti.zero_grad()  

                seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()

                logits = net(seq, attn_masks)

                loss = criterion(logits, labels)

                loss.backward()
                opti.step()

            val_acc, val_loss = self._evaluate(net, criterion, val_loader)
            tests, preds = self._evaluate_pairs(net, val_loader)
            if verbose:
                print(classification_report(tests, preds, digits=4))
                print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(ep+1, val_acc, val_loss))

    def _evaluate_pairs(self, net, dataloader):
        net.eval()
        preds = []
        tests = []
        with torch.no_grad():
            for seq, attn_masks, labels in dataloader:
                seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()
                logits = net(seq, attn_masks)
                probs = torch.sigmoid(logits)
                soft_probs = probs[:, 0] <0.5 
                preds += soft_probs.squeeze().tolist()
                tests += labels.tolist()
        return tests, preds

    def _get_accuracy_from_logits(self, logits, labels):
        probs = torch.sigmoid(logits)
        soft_probs = probs.argmax(1)
        acc = (soft_probs.squeeze() == labels).float().mean()
        return acc
    
    def _evaluate(self, net, criterion, dataloader):
        net.eval()
        mean_acc, mean_loss = 0, 0
        count = 0
        with torch.no_grad():
            for seq, attn_masks, labels in dataloader:
                seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()
                logits = net(seq, attn_masks)
                mean_loss += criterion(logits, labels).item()
                mean_acc += self._get_accuracy_from_logits(logits, labels)
                count += 1

        return mean_acc / count, mean_loss / count

    def train(self, epochs=3, lr=2e-5, criterion=nn.CrossEntropyLoss(), verbose=False):
        
        net = DecisionClassifier(self.model_name)

        criterion=criterion.cuda()

        opti = optim.Adam(net.parameters(), lr = lr)

        self._train(net, criterion, opti, self.train_loader, self.val_loader, epochs, verbose)

        self.net = net
        
        pass

    def save(self, hf_token, repo_name, username):
        """
        hf_token: https://huggingface.co/settings/token
        
        Finally:
        Save manually "cls_layer.torch" in your HF model folders: 
        https://huggingface.co/{your_username}/{repo_name}/upload/main
        """
        
        hf_login(hf_token)

        self.net.bert_layer.push_to_hub(repo_name)

        self.train_loader.dataset.tokenizer.push_to_hub(repo_name)
        
        torch.save(self.net.cls_layer, "cls_layer.torch") 
        
        api = HfApi()
        
        api.upload_file(

            path_or_fileobj="cls_layer.torch",

            path_in_repo="cls_layer.torch",

            repo_id=f"{username}/{repo_name}",

        )
        
        pass

