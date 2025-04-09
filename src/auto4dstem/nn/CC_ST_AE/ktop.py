import torch

def ktop_layer(x, num_k_sparse):
    k_top_output = x.clone()
    with torch.no_grad():
        if num_k_sparse <= x.shape[1]:
            for raw in k_top_output:
                indices = torch.topk(raw, num_k_sparse)[1].to(self.device)
                mask = torch.ones(raw.shape, dtype=bool).to(self.device)
                mask[indices] = False
                raw[mask] = 0
                raw[~mask] = 1
    return k_top_output