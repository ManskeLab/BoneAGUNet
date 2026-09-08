from pathlib import Path

import pytest

from boneagunet.config import validate_assets


def test_empty_asset_directory_reports_all_required_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as error:
        validate_assets(tmp_path)
    text = str(error.value)
    assert "models/strip" in text
    assert "atlases/mcp2/atlas_mc.nii.gz" in text
    assert "atlases/mcp3/atlas_pp.nii.gz" in text

