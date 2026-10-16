"""Check data preservation and recovery from failed MOPD data preparation."""

import importlib.util
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
spec = importlib.util.spec_from_file_location(
    "mopd_prepare_data", Path(__file__).resolve().parents[2] / "mopd/tools/prepare_nonthinking_data.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    for dataset in ("gsm8k", "math"):
        (root / dataset).mkdir(parents=True)
        for split in ("train", "test"):
            row = {
                "data_source": "openai/gsm8k" if dataset == "gsm8k" else "DigitalLearningGmbH/MATH-lighteval",
                "prompt": [{"role": "user", "content": "What is 1 + 2?"}],
                "reward_model": {"style": "rule", "ground_truth": "3"},
                "extra_info": {"split": split, "index": 0},
            }
            pq.write_table(pa.Table.from_pylist([row]), root / dataset / f"{split}.parquet")
    return root


def test_preserves_data_and_changes_only_gsm8k_instruction(source, tmp_path):
    original = {path.relative_to(source): path.read_bytes() for path in source.rglob("*.parquet")}
    output = tmp_path / "output"
    prepare.prepare_data(source, output)
    assert {path.relative_to(source): path.read_bytes() for path in source.rglob("*.parquet")} == original
    assert {path.relative_to(output) for path in output.rglob("*.parquet")} == set(original)
    for relative, raw in original.items():
        target = output / relative
        if relative.parts[0] == "math":
            assert target.read_bytes() == raw
        else:
            expected = pq.read_table(source / relative).to_pylist()
            expected[0]["prompt"][-1]["content"] += prepare.FORMAT_INSTRUCTION
            assert pq.read_table(target).to_pylist() == expected


def test_missing_input_can_be_corrected_and_retried(source, tmp_path):
    missing = source / "math/test.parquet"
    raw = missing.read_bytes()
    missing.unlink()
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        prepare.prepare_data(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".output.*"))
    missing.write_bytes(raw)
    prepare.prepare_data(source, output)
    assert (output / "math/test.parquet").read_bytes() == raw


@pytest.mark.parametrize("invalid", ["role", "duplicate", "corrupt"])
def test_invalid_input_leaves_no_partial_output(source, tmp_path, invalid):
    path = source / "gsm8k/test.parquet"
    original = path.read_bytes()
    if invalid == "corrupt":
        path.write_bytes(b"invalid parquet")
    else:
        rows = pq.read_table(path).to_pylist()
        if invalid == "role":
            rows[0]["prompt"][-1]["role"] = "assistant"
        else:
            rows[0]["prompt"][-1]["content"] += prepare.FORMAT_INSTRUCTION
        pq.write_table(pa.Table.from_pylist(rows), path)
    output = tmp_path / "output"
    with pytest.raises((ValueError, pa.ArrowInvalid)):
        prepare.prepare_data(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".output.*"))
    path.write_bytes(original)
    prepare.prepare_data(source, output)
    assert (output / "gsm8k/test.parquet").is_file()


@pytest.mark.parametrize("existing", ["directory", "file", "symlink", "source"])
def test_existing_destination_is_preserved(source, tmp_path, existing):
    output = source if existing == "source" else tmp_path / "output"
    if existing == "directory":
        output.mkdir()
        (output / "keep").write_text("existing data")
    elif existing == "file":
        output.write_text("existing data")
    elif existing == "symlink":
        output.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*.parquet")}
    with pytest.raises(FileExistsError):
        prepare.prepare_data(source, output)
    assert {path.relative_to(source): path.read_bytes() for path in source.rglob("*.parquet")} == before
    if existing == "directory":
        assert (output / "keep").read_text() == "existing data"
    elif existing == "file":
        assert output.read_text() == "existing data"
    elif existing == "symlink":
        assert output.is_symlink()


def test_write_failure_is_retryable(source, tmp_path, monkeypatch):
    output = tmp_path / "output"
    original = pq.write_table

    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    with monkeypatch.context() as patch:
        patch.setattr(prepare.pq, "write_table", fail)
        with pytest.raises(OSError, match="simulated write failure"):
            prepare.prepare_data(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".output.*"))
    assert pq.write_table is original
    prepare.prepare_data(source, output)


def test_destination_created_during_preparation_is_preserved(source, tmp_path, monkeypatch):
    output = tmp_path / "output"
    original = prepare.shutil.copyfile

    def create_destination(src, dst):
        if not output.exists():
            output.mkdir()
            (output / "keep").write_text("other process")
        return original(src, dst)

    monkeypatch.setattr(prepare.shutil, "copyfile", create_destination)
    with pytest.raises(FileExistsError):
        prepare.prepare_data(source, output)
    assert (output / "keep").read_text() == "other process"
    assert not (output / "gsm8k").exists()
    assert not list(tmp_path.glob(".output.*"))


def test_publication_failure_is_retryable(source, tmp_path, monkeypatch):
    output = tmp_path / "output"
    original = Path.rename

    def fail_second_dataset(path, target):
        if Path(target) == output / "math":
            raise OSError("simulated publication failure")
        return original(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", fail_second_dataset)
        with pytest.raises(OSError, match="simulated publication failure"):
            prepare.prepare_data(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".output.*"))
    prepare.prepare_data(source, output)
