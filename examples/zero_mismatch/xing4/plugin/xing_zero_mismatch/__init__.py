"""Activate the recipe without modifying the third-party model plugin."""


def register():
    from vllm_ascend.patch.worker.xing4 import install
    from vllm_xing4_0_plugin import register as register_model

    register_model()
    install()
