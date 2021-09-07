from src.utils.basic.io import get_file_list
import pandas as pd

class LogAnalyzer(object):
    def __init__(self, logfile: str):
        self.logfile = logfile

        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.best_epoch = None
        self.train_acc = None
        self.val_acc = None
        self.test_acc = None

    def analyze(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

        with open(self.logfile) as f:
            lines = f.readlines()
            for line in lines:
                line = line.lower()
                words = line.split()
                if "train" in line:
                    mode = "train"
                elif "val" in line:
                    mode = "val"
                elif "test" in line:
                    mode = "test"
                if "classification loss" in line:
                    loss = float(words[-1])
                    if mode == "train":
                        self.train_loss.append(loss)
                    elif mode == "val":
                        self.val_loss.append(loss)
                    elif mode == "test":
                        self.test_loss = loss
                elif "classification accuracy" in line:
                    acc = float(words[-1])
                    if mode == "train":
                        self.train_acc.append(acc)
                    elif mode == "val":
                        self.val_acc.append(acc)
                    elif mode == "test":
                        self.test_acc = acc
                elif "best model" in line:
                    self.best_epoch = int(words[-1]) - 1

        if len(self.train_acc) > 0:
            self.best_train_acc = self.train_acc[self.best_epoch]
            self.best_val_acc = self.val_acc[self.best_epoch]
        self.best_train_loss = self.train_loss[self.best_epoch]
        self.best_val_loss = self.val_loss[self.best_epoch]

    def __str__(self):
        return "LogAnalyzer(train_acc={}, val_acc={}, test_acc={})".format(self.best_train_acc, self.best_val_acc,
                                                                           self.test_acc)

def analyze_screen_results(screen_dir:str):
    results = {"target":[], "train_loss":[], "val_loss":[], "test_loss":[],
               "train_acc":[], "val_acc":[], "test_acc":[]}
    logfiles = get_file_list(screen_dir, file_type_filter=".log")
    for logfile in logfiles:
        results["target"].append(logfile.split("/")[-2])
        la = LogAnalyzer(logfile)
        la.analyze()
        results["train_loss"].append(la.best_train_loss)
        results["val_loss"].append(la.best_val_loss)
        results["test_loss"].append(la.test_loss)
        results["train_acc"].append(la.best_train_acc)
        results["val_acc"].append(la.best_val_acc)
        results["test_acc"].append(la.test_acc)
    return pd.DataFrame.from_dict(results)



