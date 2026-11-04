# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ATRL entry point. Importing this module registers ``grpo_segment``."""

from recipe.atrl import algos  # noqa: F401

from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()
