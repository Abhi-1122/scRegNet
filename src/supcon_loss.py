import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError("features must be a 2D tensor of shape (batch_size, embedding_dim)")

        labels = labels.view(-1)
        if labels.shape[0] != features.shape[0]:
            raise ValueError("labels and features batch size must match")

        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.zeros((), device=features.device, dtype=features.dtype)

        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(features.device)

        logits_mask = torch.ones_like(positive_mask) - torch.eye(batch_size, device=features.device)
        positive_mask = positive_mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0

        if not torch.any(valid):
            return torch.zeros((), device=features.device, dtype=features.dtype)

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / torch.clamp(positive_count, min=1.0)
        loss = -mean_log_prob_pos[valid].mean()
        return loss
