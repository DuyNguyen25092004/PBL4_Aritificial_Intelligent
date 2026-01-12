"""Main script to run the testing of the model(ECGNet, Resnet101).
"""

__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"


import os
import random
import argparse
import json
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn

from preprocessing.preprocess import preprocess
from utils.torch_dataloader import DataGen
from utils.metrics import Metrics, AUC, metric_summary

# Random seed
seed = 42
random.seed(seed)
np.random.seed(seed)


def epoch_run(
    model: nn.Module, dataset: DataGen, device: torch.device
) -> NDArray[np.float64]:
    """Testing of the model.

    Parameters
    ----------
    model: nn.Module
        Model to be tested.
    dataset: DataGen
        Dataset to be tested.
    device: torch.device
        Device to be used.

    Returns
    -------
    NDArray[np.float64]
        Predicted values.

    """

    model.to(device)
    model.eval()
    pred_all = []

    with torch.no_grad():
        for batch_step in tqdm(range(len(dataset)), desc="test"):
            batch_x, _ = dataset[batch_step]
            batch_x = batch_x.permute(0, 2, 1).to(device)
            pred = model(batch_x)
            pred_all.append(pred.detach().cpu().numpy())
    
    pred_all_array = np.concatenate(pred_all, axis=0)

    return pred_all_array


def test(
    model: nn.Module,
    path: str = "data/ptb",
    batch_size: int = 32,
    name: str = "imle_net",
) -> None:
    """Data preprocessing and testing of the model.

    Parameters
    ----------
    model: nn.Module
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred = epoch_run(model, test_gen, device)

    roc_score = roc_auc_score(y_test, pred, average="macro")
    acc, mean_acc = Metrics(y_test, pred)
    class_auc = AUC(y_test, pred)
    summary = metric_summary(y_test, pred)

    print(f"class wise accuracy: {acc}")
    print(f"accuracy: {mean_acc}")
    print(f"roc_score : {roc_score}")
    print(f"class wise AUC : {class_auc}")
    print(f"F1 score (Max): {summary[0]}")
    print(f"class wise precision, recall, f1 score : {summary}")

    logs = {
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
    """Main function to run the training of the model."""

    # Args parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="data/ptb", help="Ptb-xl dataset location"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ecgnet",
        help="Select the model to train. (ecgnet, resnet101)",
    )
    parser.add_argument("--batchsize", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    if args.model == "ecgnet":
        from models.ECGNet import ECGNet

        model = ECGNet()
    elif args.model == "resnet101":
        from models.resnet101 import resnet101

        model = resnet101()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    path_weights = os.path.join(os.getcwd(), "checkpoints", f"{args.model}_weights.pt")
    
    # Fixed: Added weights_only=True for Python 3.12 security
    model.load_state_dict(torch.load(path_weights, weights_only=True))

    test(model, path=args.data_dir, batch_size=args.batchsize, name=args.model)
