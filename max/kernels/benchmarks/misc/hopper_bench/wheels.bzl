# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Hopper benchmark wheel library targets."""


def hopper_bench_wheel(name):
    """Returns the Bazel target for a Hopper benchmark wheel.

    Args:
        name: The wheel name (e.g., "flashinfer", "flash-attn")

    Returns:
        The fully-qualified Bazel target label.
    """
    return "//max/kernels/benchmarks/misc/hopper_bench:{}".format(name)
