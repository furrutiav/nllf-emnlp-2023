import pandas as pd
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle

class NLLFIntergration:

    def __init__(self, root_labels, file_name_support, label_col_name, dt_max_depth):
        o = [""]
        with open(file_name_support) as f:
            o = f.readlines()
        self.support = [x.replace("\n", "").strip() for x in o]
        
        self.label_col_name = label_col_name

        self.raw_data = self.get_rawdata(root_labels)

        self.data = self.prepare_data(label_col_name)

        self._train(max_depth=dt_max_depth)

    def get_rawdata(self, root_labels):
        df = {}
        for k in ["train", "val", "test"]:
            df[k] = pd.read_excel(f"{root_labels}/{k}_nllf.xlsx", index_col=0)
        return df

    def prepare_data(self, label_col_name):
        X_train = self.raw_data["train"][self.support]
        X_val = self.raw_data["val"][self.support]
        X_test = self.raw_data["test"][self.support]

        y_train = self.raw_data["train"][label_col_name].apply(int)
        y_val = self.raw_data["val"][label_col_name].apply(int)

        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, None)
        }

    def _train(self, max_depth=5, seed=42):
        X_train, y_train = self.data["train"]

        clf = DecisionTreeClassifier(random_state=seed, max_depth=max_depth)
        clf.fit(X_train[self.support], y_train)

        self.clf = clf

        pass

    def predict(self, where):
        X = self.data[where][0]
        return self.clf.predict(X)
    
    def save_predict(self, root_labels):
        for where in self.data.keys():
            X = self.data[where][0].copy()
            y_pred = self.clf.predict(X)
            X["pred"] = y_pred
            X.to_excel(f"{root_labels}/nllf_pred_{where}.xlsx")
        pickle.dump(self.clf, open(f"{root_labels}/dt_nllf.pkl", "wb"))