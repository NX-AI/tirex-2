# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""Shared-tensor bookkeeping for the safetensors checkpoint format.

:class:`~tirex2.model.TiRex2` reuses the same tensor object under several
``state_dict`` names: the bidirectional xLSTM block assigns ``rev_cell =
fwd_cell`` (one module reachable through two attributes), and the sLSTM cell
exposes ``recurrent_kernel``/``bias`` as ``ParameterProxy`` aliases of
``_recurrent_kernel_``/``_bias_``.

``safetensors`` deliberately refuses to serialize tensors that share storage,
so a checkpoint can only store one name per alias group. This module is the
single place that decides which name survives (:func:`drop_shared_duplicates`,
used when writing) and how the remaining names are restored
(:func:`expand_shared_duplicates`, used when loading), so the two directions
cannot drift apart.

Alias groups are derived from the live module graph by tensor identity rather
than from name patterns, so a newly shared module is handled automatically.
"""

from collections import defaultdict

import torch
from torch import nn

__all__ = ["shared_tensor_groups", "drop_shared_duplicates", "expand_shared_duplicates"]


def shared_tensor_groups(model: nn.Module) -> list[tuple[str, ...]]:
    """Group the ``state_dict`` names that refer to one and the same tensor.

    Parameters
    ----------
    model : torch.nn.Module
        Instantiated model whose module graph defines the aliases.

    Returns
    -------
    list of tuple of str
        One sorted tuple per alias group, each holding at least two names.
        Groups are sorted, so the result is deterministic. Names that are not
        aliased anywhere are absent entirely.
    """
    by_identity: dict[int, list[str]] = defaultdict(list)
    named_tensors = [
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    ]
    for name, tensor in named_tensors:
        by_identity[id(tensor)].append(name)

    return sorted(tuple(sorted(names)) for names in by_identity.values() if len(names) > 1)


def drop_shared_duplicates(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Reduce every alias group in ``state_dict`` to its single canonical name.

    The surviving name is the alphabetically first one of its group. That
    choice is a convention of the writer only: :func:`expand_shared_duplicates`
    restores from whichever group member a checkpoint happens to carry.

    Parameters
    ----------
    model : torch.nn.Module
        Model defining the alias groups, see :func:`shared_tensor_groups`.
    state_dict : dict[str, torch.Tensor]
        State dict to reduce. It is not modified.

    Returns
    -------
    kept : dict[str, torch.Tensor]
        ``state_dict`` without the redundant names, insertion order preserved.
    dropped : list[str]
        The removed names, sorted.

    Raises
    ------
    ValueError
        If two names of one alias group hold different values. That means the
        state dict was not produced by this model wiring, and dropping either
        name would silently lose information.
    """
    redundant: set[str] = set()
    for group in shared_tensor_groups(model):
        present = [name for name in group if name in state_dict]
        if not present:
            continue
        canonical, *duplicates = present
        for name in duplicates:
            if not torch.equal(state_dict[canonical], state_dict[name]):
                raise ValueError(
                    f"{name!r} and {canonical!r} alias the same tensor in the model but hold "
                    f"different values in the state dict; deduplication would lose information."
                )
        redundant.update(duplicates)

    kept = {name: tensor for name, tensor in state_dict.items() if name not in redundant}
    return kept, sorted(redundant)


def expand_shared_duplicates(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Restore the alias names that :func:`drop_shared_duplicates` removed.

    Every name of an alias group with at least one entry in ``state_dict`` is
    filled in from that entry, so the result can be loaded with ``strict=True``
    and genuine key mismatches are still reported.

    Parameters
    ----------
    model : torch.nn.Module
        Model defining the alias groups, see :func:`shared_tensor_groups`.
    state_dict : dict[str, torch.Tensor]
        State dict to complete. It is not modified.

    Returns
    -------
    dict[str, torch.Tensor]
        A copy of ``state_dict`` with the missing alias names added. Groups
        that are entirely absent stay absent, so ``load_state_dict`` still
        reports them as missing keys.

    Raises
    ------
    ValueError
        If a group has several entries in ``state_dict`` that disagree; the
        checkpoint would be ambiguous about which value to load.
    """
    expanded = dict(state_dict)
    for group in shared_tensor_groups(model):
        present = [name for name in group if name in state_dict]
        if not present:
            continue
        canonical = present[0]
        for name in present[1:]:
            if not torch.equal(state_dict[canonical], state_dict[name]):
                raise ValueError(
                    f"Checkpoint holds different values for {canonical!r} and {name!r}, which "
                    f"alias the same tensor in the model."
                )
        for name in group:
            expanded.setdefault(name, state_dict[canonical])

    return expanded
