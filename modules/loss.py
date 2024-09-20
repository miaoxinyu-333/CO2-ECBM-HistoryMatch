import torch
import torch.nn.functional as F
from evaluation.eval_metric import ssim_index

def scaledlp_loss(input: torch.Tensor, target: torch.Tensor, p: int = 2, reduction: str = "mean"):
    B = input.size(0)
    diff_norms = torch.norm(input.reshape(B, -1) - target.reshape(B, -1), p, 1)
    target_norms = torch.norm(target.reshape(B, -1), p, 1)
    val = diff_norms / target_norms
    if reduction == "mean":
        return torch.mean(val)
    elif reduction == "sum":
        return torch.sum(val)
    elif reduction == "none":
        return val
    else:
        raise NotImplementedError(reduction)

class ScaledLpLoss(torch.nn.Module):
    """Scaled Lp loss for PDEs.

    Args:
        p (int, optional): p in Lp norm. Defaults to 2.
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, p: int = 2, reduction: str = "mean") -> None:
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return scaledlp_loss(input, target, p=self.p, reduction=self.reduction)


def custommse_loss(input: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
    loss = F.mse_loss(input, target, reduction="none")
    # avg across space
    reduced_loss = torch.mean(loss, dim=tuple(range(3, loss.ndim)))
    # sum across time + fields
    reduced_loss = reduced_loss.sum(dim=(1, 2))
    # reduce along batch
    if reduction == "mean":
        return torch.mean(reduced_loss)
    elif reduction == "sum":
        return torch.sum(reduced_loss)
    elif reduction == "none":
        return reduced_loss
    else:
        raise NotImplementedError(reduction)


def pearson_correlation(predictions: torch.Tensor, target: torch.Tensor, reduce_batch: bool = False):
    B = predictions.size(0)
    T = predictions.size(1)
    predictions = predictions.reshape(B, T, -1)
    target = target.reshape(B, T, -1)
    predictions_mean = torch.mean(predictions, dim=(2), keepdim=True)
    target_mean = torch.mean(target, dim=(2), keepdim=True)
    # Unbiased since we use unbiased estimates in covariance
    predictions_std = torch.std(predictions, dim=(2), unbiased=False)
    target_std = torch.std(target, dim=(2), unbiased=False)

    corr = torch.mean((predictions - predictions_mean) * (target - target_mean), dim=2) / (predictions_std * target_std).clamp(
        min=torch.finfo(torch.float32).tiny
    )  # shape (B, T)
    if reduce_batch:
        corr = torch.mean(corr, dim=0)
    return corr

class CustomMSELoss(torch.nn.Module):
    """Custom MSE loss for PDEs.

    MSE but summed over time and fields, then averaged over space and batch.

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return custommse_loss(input, target, reduction=self.reduction)


class PearsonCorrelationScore(torch.nn.Module):
    """Pearson Correlation Score for PDEs."""

    def __init__(self, channel: int = None, reduce_batch: bool = False) -> None:
        super().__init__()
        self.channel = channel
        self.reduce_batch = reduce_batch

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.channel is not None:
            predictions = predictions[:, :, self.channel]
            target = target[:, :, self.channel]
        return pearson_correlation(predictions, target, reduce_batch=self.reduce_batch)