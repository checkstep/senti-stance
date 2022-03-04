import pathlib
from abc import abstractmethod
from typing import Tuple

import pandas as pd


class BaseLoader:
    def __init__(self, dataset_path, task_name, id2label, suffix=""):
        self.dataset_path = pathlib.Path(dataset_path)
        self.task_name = task_name
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.suffix = suffix
        self.train_dataset, self.val_dataset, self.test_dataset = self._prepare_splits(
            self.dataset_path, self.task_name
        )

        for subset in (self.train_dataset, self.val_dataset, self.test_dataset):
            if subset is None:
                continue

            subset["task_name"] = task_name
            if "hypothesis" not in subset.columns:
                subset["hypothesis"] = ""

            if task_name == "wtwt":
                premise_maps = {
                    "AET_HUM": "Aetna acquires Humana",
                    "ANTM_CI": "Anthem acquires Cigna",
                    "CVS_AET": "CVS Health acquires Aetna",
                    "CI_ESRX": "Cigna acquires Express Scripts",
                    "FOXA_DIS": "Disney acquires 21st Century Fox ",
                }
                subset["premise"] = subset["premise"].apply(premise_maps.get)

        if self.train_dataset is not None:
            self.train_dataset = self.train_dataset[
                self.train_dataset.hypothesis != "[not found]"
                ].reset_index(drop=True)
        if self.val_dataset is not None:
            self.val_dataset = self.val_dataset[
                self.val_dataset.hypothesis != "[not found]"
                ].reset_index(drop=True)
        if self.test_dataset is not None:
            self.test_dataset = self.test_dataset[
                self.test_dataset.hypothesis != "[not found]"
                ].reset_index(drop=True)

    @abstractmethod
    def _prepare_splits(
            self, dataset_path: pathlib.Path, task_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass

class StanceBenchmarkLoader(BaseLoader):
    def _prepare_splits(self, dataset_path, task_name):
        train = None
        val = None
        test = None

        train_path = dataset_path / f"{task_name}_train{self.suffix}.json"
        val_path = dataset_path / f"{task_name}_dev.json"
        test_path = dataset_path / f"{task_name}_test.json"

        if train_path.exists():
            train = pd.read_json(train_path, lines=True).copy()
        if val_path.exists():
            val = pd.read_json(val_path, lines=True).copy()
        if test_path.exists():
            test = pd.read_json(test_path, lines=True).copy()

        return train, val, test


class CrossLingStanceBenchmark(StanceBenchmarkLoader):
    def _prepare_splits(self, dataset_path, task_name):
        train, val, test = super()._prepare_splits(dataset_path, task_name)

        for df in (train, val, test):
            if df is not None:
                df["label"] = df["label"].apply(
                    lambda x: self.label2id[x.lower().replace(" ", "_")]
                )

        return train, val, test


class SentimentStanceLoader(StanceBenchmarkLoader):
    def _prepare_data_split(self, df):
        import nltk
        nltk.download('punkt')

        def build_premise(row, expand_summary=False):
            from nltk import sent_tokenize

            title = row["title"].strip()
            heading = row["heading"].strip()

            premise = (
                sent_tokenize(row["summary"])[0] if expand_summary and row["summary"] else title
            )
            if heading != title:
                premise = f"{premise} {heading}"

            return premise

        df = df.rename({"sentence": "hypothesis"}, axis=1)
        df["premise"] = df.apply(build_premise, axis=1)
        df = df[df.c_xlmr_senti.notna()]

        df_long_prem = df.sample(frac=0.5).copy()
        df_long_prem["premise"] = df.apply(build_premise, axis=1, expand_summary=True)
        df = pd.concat((df_long_prem, df), ignore_index=True).reset_index().sample(frac=1.0)
        df["label"] = df["c_xlmr_senti"].str.lower()
        df["label"] = df["label"].str.replace("positive", "favor").str.replace("neutral", "discuss").str.replace(
            "negative", "against")

        df_unrelated = df.sample(frac=0.3).copy()
        df_unrelated["label"] = "unrelated"
        df_unrelated["premise"] = df["premise"].sample(len(df_unrelated)).values

        df = pd.concat((df_unrelated, df), ignore_index=True).reset_index().sample(frac=1.0)
        df["uid"] = df.index

        df["label"] = df["label"].apply(lambda x: self.label2id[x.lower().replace(" ", "_")])

        return df[["uid", "hypothesis", "premise", "label"]]

    def _prepare_splits(self, dataset_path, task_name):
        print(dataset_path)
        train, val, test = super()._prepare_splits(dataset_path, task_name)
        if train is not None:
            train = self._prepare_data_split(train)
        if val is not None:
            val = self._prepare_data_split(val)
        if test is not None:
            test = self._prepare_data_split(test)

        return train, val, test

