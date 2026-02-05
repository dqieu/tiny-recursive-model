import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from torch.utils.data import DataLoader

from tiny_recursive_model.dataio import padded_batch
from tqdm import tqdm


def binary_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "specificity": tn / (tn + fp),
        "fpr": fp / (fp + tn),
        "fnr": fn / (fn + tp),
    }


def evaluate(model, dataloader, device):
    model.eval()
    model.to(device)

    batch_results = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            y_prob, _ = model.predict(inputs)
            y_pred = (y_prob >= 0.5).long()

            batch_results.setdefault("y_true", []).extend(labels.cpu().numpy().tolist())
            batch_results.setdefault("y_pred", []).extend(y_pred.cpu().numpy().tolist())
            batch_results.setdefault("y_prob", []).extend(y_prob.cpu().numpy().tolist())

    return binary_metrics(batch_results["y_true"],
                           batch_results["y_pred"],
                           batch_results["y_prob"])
