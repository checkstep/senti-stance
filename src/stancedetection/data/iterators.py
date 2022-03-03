from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain

import nlpaug.augmenter.word as naw
import nltk
import numpy as np
import torch
from nltk.corpus import words
from pyinflect import getInflection
from torch.utils.data import Dataset

from stancedetection.models.nn_constants import SYNONYM_THESAURUS, SYNONYM_THESAURUS_MULTILINGUAL
from stancedetection.util.mappings import TASK_MAPPINGS, TGT_CTX_PREFIXES

nltk.download("words")


@dataclass
class LabelEncoderArguments:
    positive_samples_synonyms: int = 0
    negative_samples_synonyms: int = 0
    negative_samples_rand: int = 0
    p_replace_pos_label: float = 0.0
    p_replace_neg_label: float = 0.0
    p_mask: float = 0.0
    p_random: float = 0.0
    same_language_labels: bool = False
    p_swap: float = 0.1
    p_delete: float = 0.1
    p_split: float = 0.1
    p_label_cond: float = 0.0


@dataclass
class Augmenter:
    p_apply: float = 0.0
    module: naw.WordAugmenter = None


@dataclass
class Pattern:
    text: str = "The stance of the following %(HTYPE)s %(MASK)s is %(MASK)s the %(PTYPE)s %(MASK)s."

    positions: tuple = ("hypothesis", "label", "premise")


class BaseDataset(Dataset):
    def __init__(self, dataset, id2label, task2labels):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.id2label = id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.task2labels = task2labels
        self.label2task = {}
        for task, labels in enumerate(task2labels):
            self.label2task.update({label: task for label in labels})

    def __len__(self):
        return len(self.dataset)

    @abstractmethod
    def __getitem__(self, idx):
        pass


class StanceDataset(BaseDataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        max_seq_length,
        id2label,
        task2labels,
        le_args: LabelEncoderArguments,
        add_prefixes=False,
        do_sample=False,
        do_mlm=False,
        do_augment=False,
        do_label_cond=False,
        do_inflect=False,
        add_weights=False,
    ):
        super(StanceDataset, self).__init__(dataset, id2label, task2labels)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.dataset = dataset[["uid", "hypothesis", "premise", "task_name", "label"]].to_records()
        self.add_prefixes = add_prefixes
        self.do_sample = do_sample
        self.do_inflect = do_inflect
        self.add_weights = add_weights
        self.le_args = le_args

        self.do_label_cond = do_label_cond and self.le_args.p_label_cond > 0.0
        self.do_mlm = do_mlm and (self.le_args.p_mask > 0.0 or self.le_args.p_random > 0.0)
        self.words = list(words.words()) if self.le_args.negative_samples_rand > 0 else []
        self.augmenters = defaultdict(Augmenter)
        self.positive_pattern = Pattern()

        if do_augment:
            if self.le_args.p_swap > 0:
                self.augmenters["swap"] = Augmenter(
                    p_apply=0.5,
                    module=naw.RandomWordAug(
                        action="swap",
                        aug_p=self.le_args.p_swap,
                        aug_max=None,
                        aug_min=1,
                    ),
                )

            if self.le_args.p_delete > 0:
                self.augmenters["delete"] = Augmenter(
                    p_apply=0.5,
                    module=naw.RandomWordAug(
                        action="delete",
                        aug_p=self.le_args.p_delete,
                        aug_max=None,
                        aug_min=1,
                    ),
                )

            if self.le_args.p_split > 0:
                self.augmenters["split"] = Augmenter(
                    p_apply=0.5,
                    module=naw.SplitAug(
                        aug_p=self.le_args.p_split,
                        aug_max=None,
                        aug_min=1,
                        min_char=4,
                    ),
                )

    def _clean_label(self, label):
        return label.split("__")[-1].lower().replace("_", " ")

    def _sample_synonyms(self, synonyms, n_samples):
        if not synonyms or not n_samples:
            return []

        result = [synonyms[x] for x in np.random.randint(0, len(synonyms), n_samples)]
        return result

    def _sample_synonyms_label(self, label, n_samples):
        synonyms = SYNONYM_THESAURUS.get(label, [])
        return self._sample_synonyms(synonyms, n_samples)

    def _get_or_replace_label(self, label, p_replace):
        if not self.do_sample or p_replace == 0.0:
            return label
        p = np.random.random()

        new_label = label
        if p < p_replace:
            synonym = self._sample_synonyms_label(label, 1)
            if synonym:
                new_label = synonym[0]
        return new_label

    def _mask_input_ids(self, input_ids, pattern_mask, p_mask=0.0, p_random=0.0):
        special_tokens_mask = input_ids != self.tokenizer.mask_token_id

        if pattern_mask is not None:
            special_tokens_mask &= pattern_mask

        masked_indices = np.random.binomial(1, p=p_mask, size=input_ids.shape).astype(bool)
        input_ids[special_tokens_mask & masked_indices] = self.tokenizer.mask_token_id

        if p_random > 0.0:
            random_indices = np.random.binomial(1, p=p_random, size=input_ids.shape).astype(bool)
            random_indices &= special_tokens_mask & ~masked_indices
            random_words = np.random.randint(self.tokenizer.vocab_size, size=random_indices.sum())
            input_ids[random_indices] = random_words

        return input_ids

    def _get_class_weight(self, label_id):
        if not hasattr(self, "class_weights"):
            return None

        return self.class_weights[label_id]

    def _get_multilingual_label(self, label, lang):
        if lang == "en" or label == self.tokenizer.cls_token:
            return label
        return SYNONYM_THESAURUS_MULTILINGUAL[label][lang]

    def _augment(self, text, lang):
        if "zh" in lang:
            return text

        for _, augmenter in self.augmenters.items():
            if np.random.random() > augmenter.p_apply:
                continue
            text = augmenter.module.augment(text)

        return text

    def _inflect_word(self, word):
        if word == self.tokenizer.cls_token or word == self.tokenizer.sep_token_id:
            return word

        inflections = getInflection(word, "VBG", inflect_oov=False)
        if inflections:
            inflection = inflections[0]
        elif word[-3:] == "ing" or word[-3:] == " to":
            inflection = word
        else:
            inflection = f"{word} to"

        return inflection

    def _encode_pattern(
        self,
        pattern,
        premise,
        hypothesis,
        label,
        premise_type="",
        hypothesis_type="snippet",
    ):
        mask_idx = self.tokenizer.mask_token_id
        mask_token = self.tokenizer.mask_token
        pattern_text = pattern.text % {
            "HTYPE": hypothesis_type,
            "PTYPE": premise_type,
            "MASK": mask_token,
        }

        input_ids = self.tokenizer.encode(
            pattern_text, add_special_tokens=True, return_tensors="np"
        ).reshape(-1)

        # Adding 2 tokens to the budget, since we'll replace the masks,
        # but then we need to remove 2 for the [CLS] and [SEP]
        # In the end we do not need any corrections
        tokens_budget = self.max_seq_length - len(input_ids) - 1

        encoded_premise = self.tokenizer.encode(
            premise, add_special_tokens=False, max_length=tokens_budget, truncation=True
        )
        encoded_hypothesis = self.tokenizer.encode(
            hypothesis, add_special_tokens=False, max_length=tokens_budget, truncation=True
        )
        encoded_label = self.tokenizer.encode(
            label, add_special_tokens=False, max_length=tokens_budget, truncation=True
        )
        tokens_budget -= len(encoded_label)

        h_len = len(encoded_hypothesis)
        p_len = len(encoded_premise)
        while tokens_budget < h_len + p_len:
            truncation_len = h_len + p_len - tokens_budget
            if p_len == h_len:
                half_truncation_len = truncation_len // 2
                p_len -= half_truncation_len
                # If the leftover budget is uneven then we remove an extra symbol from the hypothesis
                h_len -= half_truncation_len + truncation_len % 2

            # "Longest first": We first what to make them even, and then truncate the rest equally
            elif p_len < h_len:
                h_len -= min(truncation_len, h_len - p_len)
            else:
                p_len -= min(truncation_len, p_len - h_len)

        encoded_for_mask = {
            "premise": encoded_premise[:p_len],
            "hypothesis": encoded_hypothesis[:h_len],
            "label": encoded_label,
        }

        input_ids_merged = np.zeros(
            len(input_ids) + sum(map(len, encoded_for_mask.values())) - 3, dtype=np.long
        )
        pattern_mask = np.zeros_like(input_ids_merged, dtype=bool)

        full_idx = 0
        masked_token_idx = 0
        for input_token in input_ids:
            if input_token == mask_idx:
                mask_for = pattern.positions[masked_token_idx]
                if mask_for == "label":
                    mask_token_idx = full_idx
                insert_array = encoded_for_mask[mask_for]
                insert_size = len(insert_array)
                insert_slice = slice(full_idx, full_idx + insert_size)
                input_ids_merged[insert_slice] = insert_array
                pattern_mask[insert_slice] = True
                full_idx += insert_size
                masked_token_idx += 1
            else:
                input_ids_merged[full_idx] = input_token
                full_idx += 1

        return {
            "input_ids": input_ids_merged.reshape(1, -1),
            "mask_token_idx": mask_token_idx,
            "pattern_mask": pattern_mask,
        }

    def __getitem__(self, idx):
        example = self.dataset[idx]
        label = example["label"]
        hypothesis = example["hypothesis"]
        premise = example["premise"]
        task_name = example["task_name"]
        lang = TASK_MAPPINGS[task_name].get("language", "en")
        label_name = self.id2label.get(label, label)
        correct_label_name = self._clean_label(label_name.split("__")[-1])

        if not hypothesis:
            hypothesis = premise
            premise = ""

        hypothesis = self._augment(hypothesis, lang)
        premise = self._augment(premise, lang)

        pattern_dict = {
            "premise": premise,
            "hypothesis": hypothesis,
        }

        if not self.do_label_cond or np.random.random() >= self.le_args.p_label_cond:
            pattern_dict["label"] = self.tokenizer.mask_token
        else:
            pattern_dict["label"] = correct_label_name

        if self.add_prefixes:
            pattern_dict["premise_type"] = TGT_CTX_PREFIXES[task_name]["premise"].lower()
            pattern_dict["hypothesis_type"] = TGT_CTX_PREFIXES[task_name]["hypothesis"].lower()

        encoded_input = self._encode_pattern(self.positive_pattern, **pattern_dict)
        encoded_input["attention_mask"] = np.ones_like(encoded_input["input_ids"])

        positives = [
            self._get_or_replace_label(correct_label_name, self.le_args.p_replace_pos_label)
        ]
        negatives = [
            self._get_or_replace_label(
                self._clean_label(self.id2label[id_]), self.le_args.p_replace_neg_label
            )
            for id_ in self.task2labels[self.label2task[label]]
            if id_ != label
        ]

        if self.do_sample:
            positives += self._sample_synonyms_label(
                correct_label_name, self.le_args.positive_samples_synonyms
            )

            negatives += self._sample_synonyms(self.words, self.le_args.negative_samples_rand)
            negatives += chain(
                *[
                    self._sample_synonyms_label(neg_lab, self.le_args.negative_samples_synonyms)
                    for neg_lab in negatives
                ]
            )
            negatives.append(self.tokenizer.cls_token)

        visible_task_labels = negatives + positives

        if self.le_args.same_language_labels:
            labels_for_encoding = [
                self._get_multilingual_label(x, lang) for x in visible_task_labels
            ]
        elif self.do_inflect:
            labels_for_encoding = [self._inflect_word(x) for x in visible_task_labels]
        else:
            labels_for_encoding = visible_task_labels

        encoded_input.update(
            {
                f"labels_{k}": v
                for k, v in self.tokenizer.batch_encode_plus(
                    labels_for_encoding,
                    return_token_type_ids=None,
                    return_attention_mask=True,
                    add_special_tokens=False,
                    padding="longest",
                    return_tensors="np",
                ).items()
            }
        )
        labels_count = len(visible_task_labels)
        encoded_input["labels_length"] = len(encoded_input["labels_input_ids"][0])
        encoded_input["labels"] = np.zeros(labels_count, dtype=np.long)
        encoded_input["labels"][-len(positives) :] = 1
        encoded_input["labels_mask"] = np.ones(labels_count, dtype=np.long)
        encoded_input["labels_ids"] = label
        encoded_input["labels_options"] = [f"{task_name}__{x}" for x in visible_task_labels]

        if self.add_weights:
            class_weight = self._get_class_weight(label)
            if class_weight is not None:
                encoded_input["labels_weights"] = np.full([labels_count], class_weight)

        if self.do_mlm:
            encoded_input["labels_mlm"] = encoded_input["input_ids"].copy()
            encoded_input["input_ids"] = self._mask_input_ids(
                encoded_input["input_ids"],
                pattern_mask=encoded_input["pattern_mask"],
                p_mask=self.le_args.p_mask,
                p_random=self.le_args.p_random,
            )

        encoded_input["length"] = len(encoded_input["input_ids"][0])
        encoded_input["task_name"] = task_name
        encoded_input["uid"] = example["uid"]
        encoded_input["idx"] = example["index"]
        encoded_input["true_label"] = label_name

        return encoded_input


def pad_vector(a, max_len):
    return np.pad(a, (0, max_len - len(a)), mode="constant", constant_values=0)


def collate_fn(raw_batch):
    max_len = max(x["length"] for x in raw_batch)
    max_label_tokens_len = max(x["labels_length"] for x in raw_batch)
    max_label_len = max(len(x["labels"]) for x in raw_batch)

    allowed_fields = [
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "labels",
        "labels_mask",
        "labels_input_ids",
        "labels_attention_mask",
        "labels_weights",
        "labels_mlm",
        "mask_token_idx",
    ]

    meta_fields = ["labels_ids", "task_name", "labels_options", "uid", "idx", "true_label"]

    batch = defaultdict(list)
    meta = defaultdict(list)
    for row in raw_batch:
        for field in meta_fields:
            if field not in row:
                continue
            meta[field].append(row[field])

        for field in allowed_fields:
            if field not in row:
                continue

            value = row[field]
            if isinstance(value, np.ndarray):
                if "labels" not in field or field == "labels_mlm":
                    value = pad_vector(value[0], max_len)
                elif field == "labels" or field == "labels_mask" or field == "labels_weights":
                    value = pad_vector(value, max_label_len)
                else:
                    value = np.pad(
                        value,
                        (
                            (0, max_label_len - value.shape[0]),
                            (0, max_label_tokens_len - value.shape[1]),
                        ),
                        mode="constant",
                        constant_values=0,
                    )

            batch[field].append(value)

    tensor_batch = {}

    float_fields = {"labels", "labels_weights"}

    for field in allowed_fields:
        if field not in batch:
            continue

        if field in float_fields:
            tensor_batch[field] = torch.FloatTensor(batch[field])
        else:
            tensor_batch[field] = torch.LongTensor(batch[field])

    return tensor_batch, meta
