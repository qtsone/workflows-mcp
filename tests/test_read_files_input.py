import pytest

from workflows_mcp.engine.executors_file import ReadFilesInput


def test_read_files_input_list():
    """Test that list inputs are accepted as is."""
    input_data = {"patterns": ["*.py"]}
    model = ReadFilesInput(**input_data)
    assert model.patterns == ["*.py"]


def test_read_files_input_json_string():
    """Test that JSON string inputs are parsed into a list."""
    input_data = {"patterns": '["*.py", "**/*.md"]'}
    model = ReadFilesInput(**input_data)
    assert model.patterns == ["*.py", "**/*.md"]


def test_read_files_input_invalid_json():
    """Test that invalid JSON string raises ValueError."""
    input_data = {"patterns": "{invalid-json}"}
    with pytest.raises(ValueError, match="Invalid JSON in patterns"):
        ReadFilesInput(**input_data)


def test_read_files_input_json_not_list():
    """Test that JSON that is not a list raises ValueError."""
    input_data = {"patterns": '{"key": "value"}'}
    with pytest.raises(ValueError, match="Parsed JSON must be a list"):
        ReadFilesInput(**input_data)


def test_read_files_input_empty_list():
    """Test default empty list."""
    input_data = {"path": "some/file.txt"}
    model = ReadFilesInput(**input_data)
    assert model.patterns == []
