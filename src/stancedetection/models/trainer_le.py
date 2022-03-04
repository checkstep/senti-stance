import argparse
import json
import logging
import os
import pathlib
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from transformers import AdamW, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

import wandb
from stancedetection.data import iterators as data_iterators
from stancedetection.data.iterators import LabelEncoderArguments
from stancedetection.models.label_encoder import CONCAT_STRATEGIES, XLMRobertaLabelEncoder
from stancedetection.util.mappings import ID2PREFIX, TASK_MAPPINGS
from stancedetection.util.model_utils import batch_to_device, get_learning_rate
from stancedetection.util.util import NpEncoder, configure_logging, set_seed

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_TYPES = {
    "xlm-r": XLMRobertaLabelEncoder,
}


def build_optimizer(model, total_steps, args):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * total_steps,
        num_training_steps=total_steps,
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        map_location = "cpu" if args.no_cuda else DEFAULT_DEVICE
        optimizer_path = os.path.join(args.model_name_or_path, "optimizer.pt")
        scheduler_path = os.path.join(args.model_name_or_path, "scheduler.pt")
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=map_location))
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=map_location))
        logger.info("Loaded the saved scheduler and optimizer.")
    return optimizer, scheduler


def mean_deque(deq):
    return sum(list(deq)) / len(deq)


def calc_metrics(predictions, id2label):
    y_true = predictions["true_stance_id"]
    y_pred = predictions["pred_stance_id"]
    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(id2label)))

    metrics = {}
    diag_values = conf_matrix[np.diag_indices(len(conf_matrix))]
    metrics["accuracy"] = diag_values.sum() / conf_matrix.sum()

    # Per class metrics
    clw_precision = np.nan_to_num(diag_values / conf_matrix.sum(axis=0), nan=0.0)
    clw_recall = np.nan_to_num(diag_values / conf_matrix.sum(axis=1), nan=0.0)
    clw_f1 = 2 * clw_precision * clw_recall / (clw_precision + clw_recall)
    clw_f1 = np.nan_to_num(clw_f1, nan=0.0)

    # Macro per-class average
    # metrics["precision_macro"] = clw_precision.mean()
    # metrics["recall_macro"] = clw_recall.mean()
    # metrics["f1_macro"] = clw_f1.mean()

    per_task_metrics = defaultdict(list)

    for i, true in id2label.items():
        metrics[f"precision_clw_{true}"] = clw_precision[i]
        metrics[f"recall_clw_{true}"] = clw_recall[i]
        metrics[f"f1_clw_{true}"] = clw_f1[i]

        task = true.split("__")[0]
        per_task_metrics[f"precision_task_{task}"].append(clw_precision[i])
        per_task_metrics[f"recall_task_{task}"].append(clw_recall[i])
        per_task_metrics[f"f1_task_{task}"].append(clw_f1[i])

    per_task_metrics = {k: np.mean(v) for k, v in per_task_metrics.items()}

    # Macro average
    for metric_type in ["precision", "recall", "f1"]:
        metrics[f"{metric_type}_macro"] = np.mean(
            [v for k, v in per_task_metrics.items() if metric_type in k]
        )

    metrics.update(per_task_metrics)
    metrics = json.loads(json.dumps(metrics, sort_keys=True, cls=NpEncoder))

    return metrics


def print_metrics(metrics, is_test=False):
    set_name = "Test" if is_test else "Dev"
    for metric, value in metrics.items():
        logger.info("%s %s: %.3f", set_name, metric, value)


def evaluate_and_export(model, dataset, subset_name, args):
    # Always put is_test=True to avoid shuffling
    predictions, eval_loss = evaluate(model, dataset, batch_size=args.eval_batch_size, is_test=True)
    metrics = calc_metrics(predictions, model.config.id2label)
    metrics["loss"] = eval_loss
    print_metrics(metrics, is_test="test" == subset_name)

    for metric, value in metrics.items():
        wandb.run.summary[f"summary_{subset_name}_{metric}"] = value

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    export_metrics(metrics, args.output_dir, prefix=subset_name + "_")
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(os.path.join(args.output_dir, f"{subset_name}_predictions.csv"))


def task_to_global_label(x, label2id):
    task_class_key = f"{x.task_name}__{TASK_MAPPINGS[x.task_name]['id2label'][x.label].lower()}"
    # Use the string labels if the task is out of domain, and not present in the labels
    global_label = label2id.get(task_class_key, task_class_key)

    return global_label


def print_subset_statistics(subset, id2label, subset_name):
    logger.info("----- Loaded %s %d examples. -----", subset_name, len(subset))
    label_stats = subset.label.value_counts(normalize=True)

    # Use the string labels if the task is out of domain, and not present in the labels
    label_stats.index = [i if isinstance(i, str) else id2label[i] for i in label_stats.index]
    for k, v in (label_stats * 100).round(2).items():
        logger.info("  Label:\t%s\t%.2f%%", k, v)


def create_datasets(args, tokenizer, model_config):
    datasets = defaultdict(list)
    fill_label_maps = not model_config or not hasattr(model_config, "task2id")

    if fill_label_maps:
        labelmaps = {
            "label2id": {},
            "task2id": {},
            "task2labels": [[] for _ in range(len(args.task_names))],
            "label2task": {},
        }
    else:
        labelmaps = {
            "id2label": model_config.id2label,
            "label2id": model_config.label2id,
            "task2id": model_config.task2id,
            "task2labels": model_config.task2labels,
            "label2task": model_config.label2task,
        }

    for task in args.task_names:
        task_description = TASK_MAPPINGS[task]

        task_id2label = {k: v.lower() for k, v in task_description["id2label"].items()}
        data_dir = pathlib.Path(args.data_dir) / task_description["task_dir"]
        dataset_loader = task_description["loader"](
            data_dir, task, task_id2label, suffix=args.dataset_suffix
        )

        if args.do_train:
            datasets["train"].append(dataset_loader.train_dataset)

        if args.evaluate_during_training or args.do_eval:
            datasets["val"].append(dataset_loader.val_dataset)

        if args.do_eval:
            datasets["test"].append(dataset_loader.test_dataset)

        if fill_label_maps:
            start_id = len(labelmaps["label2id"])
            task_label2id = {
                f"{task}__{lbl}": (start_id + id_) for id_, lbl in task_id2label.items()
            }
            labelmaps["label2id"].update(task_label2id)

            task_id = len(labelmaps["task2id"])
            labelmaps["task2id"][task] = task_id
            for label_id in sorted(task_label2id.values()):
                labelmaps["task2labels"][task_id].append(label_id)
                labelmaps["label2task"][label_id] = task_id

    if fill_label_maps:
        labelmaps["id2label"] = {v: k for k, v in labelmaps["label2id"].items()}

    merged_datasets = {}
    for k in list(datasets.keys()):
        subset: pd.DataFrame = pd.concat(datasets[k], axis=0)
        subset["label"] = subset.apply(task_to_global_label, axis=1, label2id=labelmaps["label2id"])
        merged_datasets[k] = subset

    logger.info("  Label to ID mappings for the tasks:")
    logger.info(json.dumps(labelmaps, sort_keys=True, indent=2))

    data_iters = {"train": None, "val": None, "test": None}

    le_args = LabelEncoderArguments(
        positive_samples_synonyms=args.positive_samples_synonyms,
        negative_samples_synonyms=args.negative_samples_synonyms,
        negative_samples_rand=args.negative_samples_rand,
        p_replace_pos_label=args.p_replace_pos_label,
        p_replace_neg_label=args.p_replace_neg_label,
        p_mask=args.p_mask,
        p_random=args.p_random,
        same_language_labels=args.same_language_labels,
        p_swap=args.p_swap,
        p_delete=args.p_delete,
        p_split=args.p_split,
        p_label_cond=args.p_label_cond,
    )

    for subset_name, merged_dataset in merged_datasets.items():
        is_train_subset = subset_name == "train"
        data_iters[subset_name] = data_iterators.StanceDataset(
            merged_dataset,
            tokenizer,
            args.max_seq_length,
            labelmaps["id2label"],
            labelmaps["task2labels"],
            le_args,
            add_prefixes=args.add_prefixes,
            do_sample=is_train_subset,
            do_mlm=is_train_subset,
            do_augment=is_train_subset,
            do_label_cond=is_train_subset,
            do_inflect=args.do_inflect,
            add_weights=is_train_subset and args.balanced,
        )
        print_subset_statistics(merged_dataset, labelmaps["id2label"], subset_name)

    return data_iters, labelmaps if fill_label_maps else None


def load_model_from_pretrained(
    model_name_or_path,
    model_config,
    model_type="auto",
    labelmaps=None,
    cache_dir=None,
    replace_classification=False,
    add_prefixes=False,
):
    from_pretrained_kwargs = {
        "config": model_config,
        "from_tf": bool(".ckpt" in model_name_or_path),
        "cache_dir": cache_dir,
    }
    model_cls = MODEL_TYPES[model_type]
    model = model_cls.from_pretrained(model_name_or_path, **from_pretrained_kwargs)

    if replace_classification:
        logger.info("  Adding task information to the model's config...")

        for k, v in labelmaps.items():
            setattr(model.config, k, v)

        if add_prefixes and model.config.type_vocab_size != len(ID2PREFIX):
            # Update config to finetune token type embeddings
            model.config.type_vocab_size = len(ID2PREFIX)

            # Create a new Embeddings layer, with 2 possible segments IDs instead of 1
            model.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(
                model.config.type_vocab_size, model.config.hidden_size
            )

            # Initialize it
            model.roberta.embeddings.token_type_embeddings.weight.data.normal_(
                mean=0.0, std=model.config.initializer_range
            )

        logger.info("  Replacing the classification layer...")

    logger.info(model.config)

    return model.to(DEFAULT_DEVICE)


def export_metrics(metrics, output_dir, prefix=""):
    logger.info("Saving metrics to %s", output_dir)
    with open(os.path.join(output_dir, f"{prefix}metrics.json"), "w") as fp:
        json.dump(metrics, fp, sort_keys=True, cls=NpEncoder, indent=2)


def export_model(model, tokenizer, optimizer, scheduler, args, metrics, global_step):
    # Save model checkpoint
    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", output_dir)
    export_metrics(metrics, output_dir)


def train(model, tokenizer, optimizer, scheduler, train_dataset, val_dataset, args):
    best_metric = -np.inf
    checkpoint_stats = {}
    loss_history = deque(maxlen=10)
    acc_history = deque(maxlen=10 * args.gradient_accumulation_steps)

    model.zero_grad()
    stop_training = False

    # Adding +1 to account for missed steps due to scaling of the loss in mixed precision
    num_epochs = args.num_train_epochs + int(args.fp16 and not args.no_cuda)
    train_iterator = trange(num_epochs, position=0, leave=True, desc="Epoch")
    scaler = GradScaler(enabled=args.fp16)

    for _ in train_iterator:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=args.train_batch_size,
            pin_memory=True,
            persistent_workers=False,
            drop_last=False,
            num_workers=0,
            collate_fn=data_iterators.collate_fn,
        )

        epoch_iterator = tqdm(
            train_dataloader,
            position=1,
            leave=True,
            desc="Iteration loss nan acc nan lr 0.0 opt. step 0",
        )

        tr_loss = 0
        for step, (batch, _) in enumerate(epoch_iterator):
            batch = batch_to_device(batch, device=DEFAULT_DEVICE)

            model.train()
            with autocast(enabled=args.fp16):
                model_outputs = model(**batch)
                loss = model_outputs.loss
                logits = model_outputs.le_logits

                loss /= args.gradient_accumulation_steps
                scaler.scale(loss).backward()

            labels = batch["labels"].detach().cpu().numpy().reshape(-1)
            logits = torch.sigmoid(logits).round().detach().cpu().numpy().reshape(-1)
            acc = (labels == logits).mean()

            # This is loss between the MLM and LE
            tr_loss += float(loss.item())

            acc_history.append(acc)

            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                model.zero_grad()

                global_step = get_optimizer_step(optimizer)
                learning_rate = get_learning_rate(optimizer)

                loss_history.append(tr_loss)
                epoch_iterator.set_description(
                    "Iteration loss {:.4f} acc {:.4f} lr {:.2e} opt. step {}".format(
                        mean_deque(loss_history),
                        mean_deque(acc_history),
                        learning_rate,
                        global_step,
                    )
                )

                accuracy = float(np.mean(list(acc_history)[-args.gradient_accumulation_steps :]))

                wandb.log(
                    {
                        "train": {
                            "loss": tr_loss,
                            "accuracy": accuracy,
                            "learning_rate": learning_rate,
                        }
                    },
                    step=global_step,
                )

                tr_loss = 0.0

            global_step = get_optimizer_step(optimizer)

            if (
                args.logging_steps > 0
                and global_step % args.logging_steps == 0
                # To avoid gradient accumulation duplicates
                and global_step not in checkpoint_stats
            ):
                best_metric, metrics = evaluate_and_compare_val(
                    model, tokenizer, optimizer, scheduler, val_dataset, best_metric, args
                )
                checkpoint_stats[global_step] = metrics

            if (global_step > args.max_steps > 0) or (abs(get_learning_rate(optimizer) < 1e-12)):
                epoch_iterator.close()
                stop_training = True
                break

        if stop_training:
            train_iterator.close()
            break

    # Check the stats on the last step
    if get_optimizer_step(optimizer) not in checkpoint_stats:
        _, metrics = evaluate_and_compare_val(
            model, tokenizer, optimizer, scheduler, val_dataset, best_metric, args
        )
        checkpoint_stats[get_optimizer_step(optimizer)] = metrics

    export_metrics(checkpoint_stats, args.output_dir, "checkpoint_")
    checkpoint_stats = [
        (k, v)
        for k, v in sorted(
            checkpoint_stats.items(), key=lambda item: -item[1][args.evaluation_metric]
        )
    ]

    logger.info("Training summary:")
    logger.info(json.dumps(checkpoint_stats, indent=2, cls=NpEncoder))

    return checkpoint_stats


def evaluate_and_compare_val(
    model, tokenizer, optimizer, scheduler, val_dataset, best_metric, args
):
    metrics = evaluate_with_metrics(model, val_dataset, args.eval_batch_size, is_test=False)
    print_metrics(metrics, is_test=False)

    global_step = get_optimizer_step(optimizer)
    wandb.log({"validation": metrics}, step=global_step)
    export_model(model, tokenizer, optimizer, scheduler, args, metrics, str(global_step))
    if best_metric < metrics[args.evaluation_metric]:
        logger.info(
            "Found new best model (%.2f old, %.2f new), exporting...",
            best_metric,
            metrics[args.evaluation_metric],
        )
        export_model(model, tokenizer, optimizer, scheduler, args, metrics, "best")
        best_metric = metrics[args.evaluation_metric]

    return best_metric, metrics


def evaluate_with_metrics(model, dataset, batch_size, is_test=False, num_workers=0):
    predictions, eval_loss = evaluate(
        model, dataset, batch_size=batch_size, is_test=is_test, num_workers=num_workers
    )
    metrics = calc_metrics(predictions, model.config.id2label)
    metrics["loss"] = eval_loss
    return metrics


def get_optimizer_step(optimizer):
    try:
        for params in optimizer.param_groups[0]["params"]:
            params_state = optimizer.state[params]
            if "step" in params_state:
                return params_state["step"]

        return -1
    except KeyError:
        return -1


@torch.no_grad()
def evaluate(
    model,
    dataset,
    batch_size,
    is_test=False,
    num_workers=0,
    max_steps=None,
):
    dataloader = DataLoader(
        dataset,
        shuffle=not is_test,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=data_iterators.collate_fn,
    )

    data_iterator = tqdm(dataloader, position=0 if is_test else 2, leave=True, desc="Evaluating")

    predictions = defaultdict(list)
    model.eval()
    eval_loss = 0
    last_step = 1
    for step, (batch, meta) in enumerate(data_iterator):
        batch = batch_to_device(batch, device=DEFAULT_DEVICE)
        if max_steps and step > max_steps:
            break

        model_outputs = model(**batch)
        loss = model_outputs.loss
        logits = model_outputs.le_logits

        eval_loss += float(loss.item())

        probs = logits.squeeze(-1).softmax(-1).detach().cpu().numpy()
        predictions["idx"] += meta["idx"]
        predictions["uid"] += meta["uid"]
        predictions["task_name"] += meta["task_name"]
        predictions["true_stance_id"] += meta["labels_ids"]
        predictions["true_stance_label"] += meta["true_label"]

        for p, opts in zip(probs, meta["labels_options"]):
            argmax_p = p.argmax()
            label_name = opts[argmax_p].replace(" ", "_")
            predictions["pred_stance_id"].append(model.config.label2id[label_name])
            predictions["pred_stance_label"].append(label_name)
            predictions["probs"].append((p[: len(opts)] * 100).round(2).tolist())
            predictions["options"].append(opts)

        last_step = step + 1

    return predictions, eval_loss / last_step


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--task_names",
        type=str,
        required=True,
        nargs="+",
        choices=TASK_MAPPINGS.keys(),
        help="The name of the task to train selected in the list: "
        + ", ".join(TASK_MAPPINGS.keys()),
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default="auto",
        choices=MODEL_TYPES.keys(),
        help="The model class that would be instantiated.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--evaluation_metric",
        type=str,
        default="f1_macro",
        choices=["f1_macro", "precision_macro", "recall_macro", "accuracy"],
        help="This metric is used to select the best model from the dev set.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the test set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0,
        type=float,
        help="Linear warmup over warmup_steps.",
    )
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument(
        "--log_on_epoch",
        action="store_true",
        help="Log every epoch. Overwrites the logging_steps parameter.",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--replace_classification",
        action="store_true",
        help="Replace the classification layer",
    )
    parser.add_argument(
        "--freeze_embeddings",
        action="store_true",
        help="Whether to freeze the embedding layer.",
    )
    parser.add_argument(
        "--freeze_layers",
        nargs="*",
        help="Whether to freeze layers with given ids.",
        type=int,
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Balance the training classes",
    )
    parser.add_argument(
        "--add_prefixes",
        action="store_true",
        help="Add Context/Target prefixes, e.g., Tweet: Some tweet",
    )
    parser.add_argument(
        "--do_inflect",
        action="store_true",
        help="Inflect the words to -ing form, or add 'to' at the end.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--dataset_suffix",
        default="",
        type=str,
        help="Suffix added to the names of the datasets.",
    )
    parser.add_argument(
        "--same_language_labels",
        action="store_true",
        help="Translate the labels to the target dataset's language.",
    )
    parser.add_argument(
        "--positive_samples_synonyms",
        default=0,
        type=int,
        help="",
    )
    parser.add_argument(
        "--negative_samples_synonyms",
        default=0,
        type=int,
        help="",
    )
    parser.add_argument(
        "--negative_samples_rand",
        default=0,
        type=int,
        help="",
    )
    parser.add_argument(
        "--p_replace_pos_label",
        default=0.0,
        type=float,
        help="",
    )
    parser.add_argument(
        "--p_replace_neg_label",
        default=0.0,
        type=float,
        help="",
    )
    parser.add_argument(
        "--p_mask",
        default=0.0,
        type=float,
        help="",
    )
    parser.add_argument(
        "--p_random",
        default=0.0,
        type=float,
        help="",
    )
    parser.add_argument(
        "--p_swap",
        default=0.0,
        type=float,
        help="",
    )
    parser.add_argument(
        "--p_delete",
        default=0.0,
        type=float,
        help="",
    )
    parser.add_argument(
        "--p_split",
        default=0.0,
        type=float,
        help="",
    )
    parser.add_argument(
        "--p_label_cond",
        default=0.0,
        type=float,
        help="",
    )
    parser.add_argument(
        "--lambda_mlm",
        default=0.0,
        type=float,
        help="",
    )
    parser.add_argument(
        "--concatenation_strategy",
        default=None,
        type=str,
        choices=CONCAT_STRATEGIES.keys(),
        help="",
    )
    args = parser.parse_args()

    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Remove duplicates and sort alphabetically
    args.task_names = sorted(list(set(args.task_names)))

    if args.no_cuda:
        global DEFAULT_DEVICE
        DEFAULT_DEVICE = "cpu"

    set_seed(args.seed)
    configure_logging()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir,
        use_fast=False,
    )

    model_config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path, cache_dir=args.cache_dir
    )
    if args.concatenation_strategy is not None:
        model_config.concatenation_strategy = args.concatenation_strategy
    model_config.lambda_mlm = args.lambda_mlm

    logger.info("***** Loading the dataset *****")

    data_iters, labelmaps = create_datasets(
        args, tokenizer, model_config if not args.replace_classification else None
    )

    model = load_model_from_pretrained(
        args.model_name_or_path,
        model_config,
        model_type=args.model_type,
        labelmaps=labelmaps,
        cache_dir=args.cache_dir,
        replace_classification=args.replace_classification,
        add_prefixes=args.add_prefixes,
    )

    if args.balanced and args.do_train:
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=range(len(labelmaps["id2label"])),
            y=pd.DataFrame(data_iters["train"].dataset).label.values,
        )

        logger.info("Loaded weights %s", class_weights.round(1).tolist())
        data_iters["train"].class_weights = class_weights  # torch.Tensor().to(DEFAULT_DEVICE)

    logger.info("Training/evaluation parameters %s", args)

    wandb.login()
    wandb.init(config=args, mode="disabled")

    if args.do_train:
        wandb.watch(model)
        steps_per_epoch = (
            int(np.ceil(len(data_iters["train"]) / args.train_batch_size))
            // args.gradient_accumulation_steps
        )
        total_steps = (
            steps_per_epoch * args.num_train_epochs if args.max_steps < 1 else args.max_steps
        )
        optimizer, scheduler = build_optimizer(model, total_steps, args)

        if args.log_on_epoch:
            args.logging_steps = steps_per_epoch

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(data_iters["train"]))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size * args.gradient_accumulation_steps,
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", total_steps)
        logger.info("  Logging steps = %d ", args.logging_steps)

        checkpoint_stats = train(
            model,
            tokenizer,
            optimizer,
            scheduler,
            data_iters["train"],
            data_iters["val"],
            args,
        )

        if checkpoint_stats:
            model_id = "best"
            best_model_path = os.path.join(args.output_dir, "checkpoint-{}".format(model_id))
            logger.info("Loading best model on validation from %s", best_model_path)

            model_config = AutoConfig.from_pretrained(
                best_model_path,
                cache_dir=args.cache_dir,
            )

            model = load_model_from_pretrained(
                best_model_path,
                model_config,
                model_type=args.model_type,
                cache_dir=args.cache_dir,
                replace_classification=False,
            )

    if args.do_eval:
        evaluate_and_export(model, data_iters["val"], "val", args)
        evaluate_and_export(model, data_iters["test"], "test", args)


if __name__ == "__main__":
    main()
