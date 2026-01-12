"""Main script to test the trained model(imle_net, mousavi, rajpurkar).
"""

__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"

import argparse
import os
import json
from typing import Any
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf

from preprocessing.preprocess import preprocess
from utils.dataloader import DataGen
from utils.metrics import Metrics, AUC, metric_summary


def test(
    model: tf.keras.Model,
    path: str = "data/ptb",
    batch_size: int = 32,
    name: str = "imle_net",
) -> None:
    """Testing the model and logging metrics.

    Parameters
    ----------
    model: tf.keras.Model
        Model to be trained.
    path: str, optional
        Path to the directory containing the data. (default: 'data/ptb')
    batch_size: int, optional
        Batch size. (default: 32)
    name: str, optional
        Name of the model. (default: 'imle_net')

    """

    _, _, X_test_scale, y_test, _, _ = preprocess(path=path)
    test_gen = DataGen(X_test_scale, y_test, batch_size=batch_size)

    pred_all = []
    gt_all = []

    for X, y in tqdm(test_gen, desc="Testing Model"):
        pred = model.predict(X, verbose=0)
        pred_all.extend(pred.tolist())
        gt_all.extend(y.tolist())

    pred_all, gt_all = np.array(pred_all), np.array(gt_all)
    roc_score = roc_auc_score(gt_all, pred_all, average="macro")
    acc, mean_acc = Metrics(gt_all, pred_all)
    class_auc = AUC(gt_all, pred_all)
    summary = metric_summary(gt_all, pred_all)

    print(f"class wise accuracy: {acc}")
    print(f"accuracy: {mean_acc}")
    print(f"roc_score : {roc_score}")
    print(f"class wise AUC : {class_auc}")
    print(f"F1 score (Max): {summary[0]}")
    print(f"class wise precision, recall, f1 score : {summary}")

    logs: dict[str, Any] = {
        "roc_score": float(roc_score),
        "mean_acc": float(mean_acc),
        "accuracy": [float(x) for x in acc],
        "class_auc": [float(x) for x in class_auc],
        "F1 score (Max)": float(summary[0]),
        "class_precision_recall_f1": summary,
    }
    logs_path = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_path, exist_ok=True)

    with open(os.path.join(logs_path, f"{name}_test_logs.json"), "w") as json_file:
        json.dump(logs, json_file, indent=4)


if __name__ == "__main__":
    """Main function to test the trained model."""

    # Args parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="data/ptb", help="Ptb-xl dataset location"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="imle_net",
        help="Select the model to test. (imle_net, mousavi, rajpurkar)",
    )
    parser.add_argument("--batchsize", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    if args.model == "imle_net":
        from models.IMLENet import build_imle_net
        from configs.imle_config import Config

        model = build_imle_net(Config())
    elif args.model == "mousavi":
        from models.mousavi import build_mousavi
        from configs.mousavi_config import Config

        model = build_mousavi(Config())
    elif args.model == "rajpurkar":
        from models.rajpurkar import build_rajpurkar
        from configs.rajpurkar_config import params

        model = build_rajpurkar(**params)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    path_weights = os.path.join(os.getcwd(), "checkpoints", f"{args.model}_weights.h5")
    model.load_weights(path_weights)

    test(model, path=args.data_dir, batch_size=args.batchsize, name=args.model)
