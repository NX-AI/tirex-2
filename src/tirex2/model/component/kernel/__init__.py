# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""sLSTM and mLSTM recurrence kernels. Import the modules directly; the triton ones need triton."""

from functools import cache


@cache
def triton_available() -> bool:
    """Whether the Triton kernels can be used; PyTorch's Windows wheels, for one, ship without Triton."""
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True
