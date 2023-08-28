import pandas
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier


class NLLFIntergration:

    def __init__(self, root_labels, support, label_col_name):
        self.support = support
        self.label_col_name = label_col_name

        self.raw_data = self.get_rawdata(root_labels)

        self.data = self.prepare_data(label_col_name)

        self._train()

    def get_rawdata(self, root_labels):
        df = {}
        for k in ["train", "val", "test"]:
            df[k] = pd.read_excel(f"{root_labels}/{k}_nllf.xlsx", index_col=0)
        return df

    def prepare_data(self, label_col_name):
        X_train = df["train"][self.support]
        X_val = df["val"][self.support]
        X_test = df["test"][self.support]

        y_train = df["train"][label_col_name].apply(int)
        y_val = df["val"][label_col_name].apply(int)

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

