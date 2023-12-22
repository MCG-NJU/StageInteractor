import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F


def MHAQ(sample_points: torch.Tensor,
         value: torch.Tensor,
         weight=None,
         n_points=1):
    B, Hq, Wq, n_heads_points, _ = sample_points.shape
    B, Ck, Hk, Wk = value.shape

    n_heads = n_heads_points//n_points

    sample_points = sample_points.view(B, Hq, Wq, n_heads, n_points, 2) \
        .permute(0, 3, 1, 2, 4, 5).contiguous().flatten(0, 1)
    sample_points = sample_points.flatten(2, 3)
    sample_points = sample_points*2.0-1.0
    value = value.view(B*n_heads, Ck//n_heads, Hk, Wk)
    out = F.grid_sample(
        value, sample_points,
        mode='bilinear', padding_mode='zeros', align_corners=False,
    )

    if weight is not None:
        weight = weight.view(B, Hq, Wq, n_heads, n_points) \
            .permute(0, 3, 1, 2, 4).flatten(0, 1).flatten(2, 3).unsqueeze(1)
        out *= weight

    return out.view(B, n_heads, Ck//n_heads, Hq, Wq, n_points).permute(0, 2, 1, 5, 3, 4)


def translate_to_linear_weight(ref: torch.Tensor, num_total, tau=2.0):
    grid = torch.arange(num_total, device=ref.device, dtype=ref.dtype).view(
        *[len(ref.shape)*[1, ]+[-1, ]])

    ref = ref.unsqueeze(-1).clone()
    l2 = (ref-grid).pow(2.0).div(tau).abs().neg()
    weight = torch.softmax(l2, dim=-1)

    return weight


def SAMPLE3D(sample_points: torch.Tensor,
             values: torch.Tensor,
             featmap_strides,
             n_points: int = 1,
             num_levels: int = None,
             mapping_stride=3.0,
             tau=2.0,
             ):
    B, Hq, Wq, n_heads_points, _ = sample_points.shape
    B, C, _, _ = values[0].shape

    n_heads = n_heads_points//n_points

    if num_levels is None:
        num_levels = len(values)

    sample_points_xy = sample_points[..., 0:2]
    sample_points_lvl = sample_points[..., 2].clone()
    sample_points_lvl_mapped = sample_points_lvl-mapping_stride
    sample_points_lvl_weight = translate_to_linear_weight(
        sample_points_lvl_mapped, num_levels, tau=tau)

    sample_points_lvl_weight_list = sample_points_lvl_weight.unbind(-1)

    out = sample_points.new_zeros(B, C//n_heads, n_heads, n_points, Hq, Wq)

    for i in range(num_levels):
        value = values[i]
        lvl_weights = sample_points_lvl_weight_list[i]

        stride = featmap_strides[i]
        mapping_size = value.new_tensor(
            [value.size(3), value.size(2)]).view(1, 1, 1, 1, -1) * stride
        normalized_xy = sample_points_xy/mapping_size

        out += MHAQ(normalized_xy, value,
                    weight=lvl_weights, n_points=n_points)

    return out, None
