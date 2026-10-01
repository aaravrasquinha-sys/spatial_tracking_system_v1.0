import json

from poi_perception.runtime.masks import MaskRegion, MaskSet, empty_mask_set, load_masks


def test_bbox_centroid_inside_polygon_is_masked():
    region = MaskRegion(name="mirror", polygon=[(400, 50), (620, 50), (620, 300), (400, 300)])
    ms = MaskSet([region])
    assert ms.is_masked((450, 100, 500, 200))  # centroid (475, 150) is inside
    assert ms.masked_region_name((450, 100, 500, 200)) == "mirror"


def test_bbox_centroid_outside_polygon_is_not_masked():
    region = MaskRegion(name="mirror", polygon=[(400, 50), (620, 50), (620, 300), (400, 300)])
    ms = MaskSet([region])
    assert not ms.is_masked((0, 0, 60, 100))  # centroid (30, 50) is well outside
    assert ms.masked_region_name((0, 0, 60, 100)) is None


def test_empty_mask_set_masks_nothing():
    ms = empty_mask_set()
    assert not ms.is_masked((10, 10, 20, 20))


def test_load_masks_keyed_by_cam_id(tmp_path):
    payload = {
        "cam0": [{"name": "tv", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}],
        "cam1": [],
    }
    path = tmp_path / "masks.json"
    path.write_text(json.dumps(payload))

    all_masks = load_masks(path)
    assert set(all_masks.keys()) == {"cam0", "cam1"}
    assert all_masks["cam0"].is_masked((2, 2, 8, 8))
    assert not all_masks["cam1"].is_masked((2, 2, 8, 8))
