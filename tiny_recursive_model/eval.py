import torch
from tqdm import tqdm


@torch.inference_mode()
def binary_metrics_from_tensors(y_true: torch.Tensor,
                                y_prob: torch.Tensor,
                                threshold: float = 0.5,
                                eps: float = 1e-12):
    """
    y_true: (N,) int/bool tensor with values in {0,1}
    y_prob: (N,) float tensor with values in [0,1]
    Returns scalar tensors (same device as inputs).
    """
    y_true = y_true.to(torch.int64).view(-1)
    y_prob = y_prob.to(torch.float32).view(-1)
    y_pred = (y_prob >= threshold).to(torch.int64)

    # Confusion terms (no sklearn, all tensor)
    tp = ((y_pred == 1) & (y_true == 1)).sum().to(torch.float32)
    tn = ((y_pred == 0) & (y_true == 0)).sum().to(torch.float32)
    fp = ((y_pred == 1) & (y_true == 0)).sum().to(torch.float32)
    fn = ((y_pred == 0) & (y_true == 1)).sum().to(torch.float32)

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    specificity = tn / (tn + fp + eps)
    fpr = fp / (fp + tn + eps)
    fnr = fn / (fn + tp + eps)

    roc_auc = binary_roc_auc(y_true, y_prob)  # scalar tensor

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
    }


def binary_roc_auc(y_true: torch.Tensor, y_score: torch.Tensor, eps: float = 1e-12):
    """
    Pure-torch ROC AUC using rank statistics (Mann–Whitney U).
    Works on GPU. Returns 0.5 if only one class is present.
    """
    y_true = y_true.to(torch.int64).view(-1)
    y_score = y_score.to(torch.float32).view(-1)

    n_pos = (y_true == 1).sum().to(torch.float32)
    n_neg = (y_true == 0).sum().to(torch.float32)

    # If AUC is undefined (no positives or no negatives), return 0.5 (common default)
    if (n_pos < 1) or (n_neg < 1):
        return torch.tensor(0.5, device=y_true.device, dtype=torch.float32)

    # ranks of y_score (average rank for ties via stable sort trick isn't perfect;
    # this is the common boilerplate approximation using dense ordering).
    order = torch.argsort(y_score)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, y_score.numel() + 1, device=y_score.device, dtype=torch.float32)

    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg + eps)
    return auc.clamp(0.0, 1.0)


@torch.inference_mode()
def evaluate(model, dataloader, device, threshold: float = 0.5):
    model.eval()
    model.to(device)

    y_true_chunks = []
    y_prob_chunks = []

    for batch in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
        inputs, labels = batch
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Expect: y_prob in [0,1], shape (B,) or (B,1)
        y_prob, _ = model.predict(inputs)
        y_prob = y_prob.view(-1)

        y_true_chunks.append(labels.view(-1))
        y_prob_chunks.append(y_prob)

    y_true = torch.cat(y_true_chunks, dim=0)
    y_prob = torch.cat(y_prob_chunks, dim=0)

    metrics = binary_metrics_from_tensors(y_true, y_prob, threshold=threshold)

    # Convert to Python floats for logging/printing (optional)
    metrics = {k: float(v.detach().cpu()) for k, v in metrics.items()}
    return metrics