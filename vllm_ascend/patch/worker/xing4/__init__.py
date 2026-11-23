"""Explicitly enabled Xing4 training/inference arithmetic adapters.

The recipe's vLLM general-plugin entry point calls this after the upstream
Xing model plugin has registered. Importing vllm-ascend alone does not enable it.
"""


def install():
    from . import activation, attention, linear, mhc, moe, rotary, router

    mhc.install()
    attention.install()
    rotary.install()
    linear.install()
    activation.install()
    moe.install()
    moe.install_activation()
    router.install()
