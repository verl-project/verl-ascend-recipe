# verl-ascend-recipe

`verl--ascend-recipe` hosts ascend specified recipes based on [verl](https://github.com/verl-project/verl) contributed by the community.

## Usage

`verl-recipe` can be used as a submodule of `verl`:

```bash
git clone https://github.com/verl-project/verl.git
git clone https://github.com/verl-project/verl-ascend-recipe.git
mkdir -p verl/verl_ascend_recipe && cp -r verl-ascend-recipe/* verl/verl_ascend_recipe
cd verl
```

## Required `verl` version per recipe

Every recipe directory ships a small **`REQUIRED_VERL.txt`** next to its `README.md` (same filename everywhere). That file is the canonical place for:

- upstream git URL (today [`verl-project/verl`](https://github.com/verl-project/verl), historically `volcengine/verl`),
- whether the recipe **tracks `main`** (`pip install -e .` from the same tree) or **pins** a git commit / release tag,
- a copy-pastable `pip install …` line when a pin exists.

Each recipe `README.md` links to its `REQUIRED_VERL.txt` in a short **Required `verl` version** section. The repository root [`README.md`](../README.md) also points here for discoverability.

### One-shot installer: [`install_verl.sh`](install_verl.sh)

`install_verl.sh` reads a recipe's `REQUIRED_VERL.txt` and installs the pinned core `verl` for you. It understands every `MODE=` value listed above and prints the command it will run before executing it (use `--show` for a dry-run).

```bash
# From the root of this repo (the directory containing install_verl.sh):

# List every recipe + its pinned `pip install` line
./install_verl.sh --list

# Dry-run: see exactly which pip command a recipe would execute
./install_verl.sh --recipe dapo --show

# Install the pinned verl core for a recipe via pip (default)
./install_verl.sh --recipe retool

# Recipes that expose multiple variants: pick one with --option
./install_verl.sh --recipe dapo   --option reproduction   # DAPO paper SHA
./install_verl.sh --recipe flowrl --option A              # FlowRL v0.4.0 tag
./install_verl.sh --recipe spin   --option baseline       # SPIN v0.3.0.post1
```

Flags: `--recipe NAME` (e.g. `gkd/megatron`, `specRL/histoSpec`) or `--file PATH/TO/REQUIRED_VERL.txt`, `--method pip|git` (default `pip`), `--dest DIR` (default `./verl`, only used with `--method git`), `--show` (dry-run), `--yes` (skip confirmation), `--list`, `--help`.

The script requires only `bash`, `git`, `awk`, and `pip`/`pip3` on `PATH`. It does **not** `source` or `eval` the `REQUIRED_VERL.txt` file — values are extracted with `awk` and the exact command to be executed is echoed before it runs.

| Recipe | `REQUIRED_VERL.txt` |
| --- | --- |
| dapo | [`recipe/dapo/REQUIRED_VERL.txt`](dapo/REQUIRED_VERL.txt) |
| dppo | [`recipe/dppo/REQUIRED_VERL.txt`](dppo/REQUIRED_VERL.txt) |
| deepeyes | [`recipe/deepeyes/REQUIRED_VERL.txt`](deepeyes/REQUIRED_VERL.txt) |
| flash_rl_ascend | [`recipe/flash_rl_ascend/REQUIRED_VERL.txt`](flash_rl_ascend/REQUIRED_VERL.txt) |
| r1_ascend | [`recipe/r1_ascend/REQUIRED_VERL.txt`](r1_ascend/REQUIRED_VERL.txt) |
| retool | [`recipe/retool/REQUIRED_VERL.txt`](retool/REQUIRED_VERL.txt) |

## Contribution

### Version Specification

Add or update **`REQUIRED_VERL.txt`** whenever a recipe gains a new tested pin or intentionally moves forward on `main`. Examples of valid `pip` forms:

```
# release version
verl==0.6.0

# dev version
verl@git+https://github.com/verl-project/verl.git@313dfdb2199124a37189e32e6d4a6c654379f2d4
```

### Code Linting and Formatting

To maximize flexiblility but minimize meaningless changes, we apply `pre-commit` but only force code linting and formatting with `ruff`. Use it as follows:

```bash
pip install pre-commit
pre-commit install
# for staged changes
pre-commit run
# for all files in the repo
pre-commit run --all-files
```
