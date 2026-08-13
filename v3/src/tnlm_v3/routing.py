"""Route validation and lane accounting for the V3 streaming model.

Route semantics are intentionally distinct from padding semantics:

* ``NULL_ROUTE`` (``-1``) at a valid position is a read-only/query event. It
  advances the valid-event clock but updates no routed lane.
* Routes ``0`` through ``branches - 1`` select local branch lanes.
* Route ``branches`` selects the dedicated global lane.
* A position whose ``valid_mask`` is false is padding and is a total no-op,
  including no clock advance. Its route value is ignored and may be garbage.

The helpers in this module only validate and account for routed lanes. The
caller owns the valid-event clock, including advancement for valid null events.
"""

from __future__ import annotations

import operator

import torch


NULL_ROUTE = -1

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _validate_branches(branches: int) -> int:
    if isinstance(branches, bool):
        raise TypeError("branches must be a positive integer")
    try:
        value = operator.index(branches)
    except TypeError as exc:
        raise TypeError("branches must be a positive integer") from exc
    if value < 1:
        raise ValueError("branches must be positive")
    return value


def validate_routes(
    routes: torch.Tensor,
    valid_mask: torch.Tensor,
    branches: int,
) -> None:
    """Validate a batch of route IDs without inspecting padded route values.

    Args:
        routes: Integer tensor shaped ``[batch, time]``.
        valid_mask: Boolean tensor with the same shape as ``routes``.
        branches: Number of local branches. The route equal to this value is
            reserved for the dedicated global lane.

    Raises:
        TypeError: If inputs have unsupported types or dtypes.
        ValueError: If shapes, ranks, devices, branch count, or valid route IDs
            violate the routing contract.
    """

    branch_count = _validate_branches(branches)
    if not isinstance(routes, torch.Tensor):
        raise TypeError("routes must be a torch.Tensor")
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid_mask must be a torch.Tensor")
    if routes.ndim != 2:
        raise ValueError("routes and valid_mask must have shape [batch, time]")
    if routes.shape != valid_mask.shape:
        raise ValueError("routes and valid_mask must have the same shape")
    if routes.device != valid_mask.device:
        raise ValueError("routes and valid_mask must be on the same device")
    if routes.dtype not in _INTEGER_DTYPES:
        raise TypeError("routes must use an integer dtype")
    if valid_mask.dtype is not torch.bool:
        raise TypeError("valid_mask must use torch.bool")

    valid_routes = routes[valid_mask]
    if valid_routes.numel() == 0:
        return
    in_range = (valid_routes >= NULL_ROUTE) & (valid_routes <= branch_count)
    if not bool(in_range.all()):
        invalid = valid_routes[~in_range]
        sample = invalid[:8].detach().cpu().tolist()
        raise ValueError(
            f"valid route IDs must be in [{NULL_ROUTE}, {branch_count}]; "
            f"found {sample}"
        )


def routed_lane_mask(
    routes: torch.Tensor,
    valid_mask: torch.Tensor,
    branches: int,
) -> torch.Tensor:
    """Return a boolean ``[batch, time, branches + 1]`` routed-lane mask.

    Valid null events and padded positions produce all-false rows. The final
    lane is the dedicated global lane.
    """

    validate_routes(routes, valid_mask, branches)
    branch_count = _validate_branches(branches)
    lane_ids = torch.arange(
        branch_count + 1,
        dtype=routes.dtype,
        device=routes.device,
    )
    return (routes.unsqueeze(-1) == lane_ids) & valid_mask.unsqueeze(-1)


def route_counts(
    routes: torch.Tensor,
    valid_mask: torch.Tensor,
    branches: int,
) -> torch.Tensor:
    """Count routed updates per sample, including the dedicated global lane."""

    return routed_lane_mask(routes, valid_mask, branches).sum(
        dim=1,
        dtype=torch.int64,
    )
