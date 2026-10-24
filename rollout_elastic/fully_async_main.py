"""Run the patched canonical fully-async entrypoint without redefining its actors."""

# Importing this package installs the recipe before the canonical main is used.
from verl.experimental.fully_async_policy.fully_async_main import main

if __name__ == "__main__":
    main()
