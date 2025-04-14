import torch

def ktop_layer(x: torch.Tensor, num_k_sparse: int, device: torch.device) -> torch.Tensor:
    """
    Apply k-top sparse transformation to the input tensor.

    This function takes an input tensor and sets all but the top k values in each row to 0.
    The top k values are set to 1. The transformation is applied in-place.

    Args:
        x (torch.Tensor): The input tensor to be transformed.
        num_k_sparse (int): The number of top values to retain in each row.
        device (torch.device): The device to perform computations on (e.g., CPU or GPU).

    Returns:
        torch.Tensor: The transformed tensor with only the top k values retained in each row.
    """
    k_top_output = x.clone()
    with torch.no_grad():
        if num_k_sparse <= x.shape[1]:
            for raw in k_top_output:
                indices = torch.topk(raw, num_k_sparse)[1].to(device)
                mask = torch.ones(raw.shape, dtype=bool).to(device)
                mask[indices] = False
                raw[mask] = 0
                raw[~mask] = 1
    return k_top_output