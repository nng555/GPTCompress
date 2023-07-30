import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Transpose(nn.Module):
    def forward(self, x):
        return torch.transpose(x, 1, 2)

def length_transformation(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    deltas: torch.Tensor,
    sigma: torch.Tensor,
    arange: Optional[torch.FloatTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Functional form of the length transformation mechanism

    Parameters
    ----------
    x: torch.Tensor
        Input representation
    x_mask: torch.Tensor
        Padding mask
    deltas: torch.Tensor
        Length delta changes
    sigma: torch.Tensor
        Bandwidth parameter (could be a nn.Parameter or torch.Tensor)

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Output representation, output padding mask, and output weight
    """
    x_mask = x_mask.bool()
    batch_size, max_len_in, d_model = x.size()
    deltas = deltas.reshape(-1)
    # compute input lengths

    encoder_mask = ~x_mask
    lens = encoder_mask.sum(-1).float()
    changed_lens = (lens + deltas.reshape(-1)).float()

    # max length
    max_len_out = int(torch.max(changed_lens))

    # outward proxy
    # tile [0, ..., Lmax] batch_size times
    if arange is None:
        arange_l = torch.arange(0, max_len_out, dtype=torch.float).tile(batch_size, 1).type_as(x)
        arange_z = (
            torch.arange(0, max_len_in, dtype=torch.float)
            .tile(batch_size, max_len_out, 1)
            .type_as(x)
        )
    else:
        arange_l = arange[:max_len_out].tile(batch_size, 1)
        arange_z = arange[:max_len_in].tile(batch_size, max_len_out, 1)

    # (batch_size, Lmax)
    mu = arange_l * lens[:, None] / changed_lens[:, None]

    # KC: in this way, we don’t get nan when learning sigma with the perfectly correct attention
    z_prime_mask = arange_l < changed_lens[:, None]  # new antimask
    padding_mask = ~z_prime_mask

    logits = -1.0 * torch.pow(arange_z - mu[:, :, None], 2) / (2.0 * torch.exp(sigma))

    exp_logits = torch.exp(logits)
    exp_logits = exp_logits.masked_fill(x_mask.unsqueeze(1), 0.0)
    exp_logits = exp_logits.masked_fill(padding_mask.unsqueeze(-1), 0.0)
    denom = exp_logits.sum(dim=2, keepdim=True)
    exp_logits = exp_logits.contiguous()
    denom = denom.masked_fill(denom == 0.0, 1.0)
    weight = exp_logits / denom

    z_prime = torch.matmul(weight, x)
    return z_prime, padding_mask, weight

