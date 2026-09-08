from pathlib import Path

import SimpleITK as sitk

from boneagunet import pipeline


def test_pipeline_runs_both_bone_branches(monkeypatch, tmp_path: Path):
    source = tmp_path / "patient mcp2.nii.gz"
    output = tmp_path / "erosions.nii.gz"
    image = sitk.Image((4, 5, 6), sitk.sitkInt16)
    image.SetSpacing((0.061, 0.061, 0.061))
    sitk.WriteImage(image, str(source))
    assets = tmp_path / "assets"
    assets.mkdir()
    calls = {"models": [], "bones": []}

    monkeypatch.setattr(pipeline, "resolve_asset_root", lambda _: assets)
    monkeypatch.setattr(pipeline.shutil, "which", lambda _: "/bin/tool")

    def fake_predict(model, inputs, output_dir, asset_root, device, case):
        calls["models"].append(model)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = output_dir / f"{case}.nii.gz"
        sitk.WriteImage(image, str(result))
        return result

    def fake_strip(image_path, label_path, output_dir, name):
        result = {}
        for key in ("stripped", "mc_mask", "pp_mask"):
            path = output_dir / f"{key}.nii.gz"
            sitk.WriteImage(image, str(path))
            result[key] = path
        return result

    def fake_register(image_path, mask_path, atlas_path, output_path, work_dir, threads):
        calls["bones"].append(Path(atlas_path).name)
        sitk.WriteImage(image, str(output_path))

    monkeypatch.setattr(pipeline, "predict", fake_predict)
    monkeypatch.setattr(pipeline, "apply_strip", fake_strip)
    monkeypatch.setattr(pipeline, "register_atlas", fake_register)
    monkeypatch.setattr(pipeline.candidates, "segment_erosions", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "predict_directory", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "combine_predictions", lambda *a, **k: 0)

    result = pipeline.run(source, output, mcp=2, assets=assets,
                          work_dir=tmp_path / "work")
    assert calls["models"] == ["strip", "edge", "closed_edge"]
    assert calls["bones"] == ["atlas_mc.nii.gz", "atlas_pp.nii.gz"]
    assert result["erosions"] == 0
    assert result["native_spacing"] == (0.061, 0.061, 0.061)
