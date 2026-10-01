from poi_perception.io.detection2d import Detection2D, JsonlDetectionWriter, read_jsonl


def _sample_det(track_id=1, frame_id=0):
    return Detection2D(
        t_capture=1.0 + frame_id * 0.033,
        frame_id=frame_id,
        cam_id="cam0",
        track_id_2d=track_id,
        bbox=(10.0, 20.0, 70.0, 190.0),
        det_conf=0.87,
        keypoints={"left_ankle": (30.0, 185.0, 0.6), "right_ankle": (50.0, 187.0, 0.65)},
        footpoint_px=(40.0, 186.0),
        footpoint_source="ankles",
        torso_polygon_px=[(20.0, 60.0), (60.0, 60.0), (55.0, 120.0), (25.0, 120.0)],
        torso_quality="full",
        low_confidence=False,
        track_state="tracked",
    )


def test_to_dict_from_dict_roundtrip():
    d = _sample_det()
    d2 = Detection2D.from_dict(d.to_dict())
    assert d2 == d


def test_jsonl_writer_reader_roundtrip(tmp_path):
    path = tmp_path / "detections.jsonl"
    writer = JsonlDetectionWriter(path)
    writer.write_frame(1.0, 0, [_sample_det(track_id=1, frame_id=0)])
    writer.write_frame(1.033, 1, [])  # an empty frame must still be logged (Section 8)
    writer.write_frame(1.066, 2, [_sample_det(track_id=1, frame_id=2), _sample_det(track_id=2, frame_id=2)])
    writer.close()

    rows = list(read_jsonl(path))
    assert len(rows) == 3
    t0, fid0, dets0 = rows[0]
    assert fid0 == 0
    assert len(dets0) == 1
    assert dets0[0].track_id_2d == 1

    t1, fid1, dets1 = rows[1]
    assert fid1 == 1
    assert dets1 == []  # empty frame preserved, not skipped

    t2, fid2, dets2 = rows[2]
    assert len(dets2) == 2
    assert {d.track_id_2d for d in dets2} == {1, 2}


def test_jsonl_rotation(tmp_path):
    path = tmp_path / "detections.jsonl"
    writer = JsonlDetectionWriter(path, rotate_max_bytes=200)
    for i in range(50):
        writer.write_frame(float(i), i, [_sample_det(track_id=1, frame_id=i)])
    writer.close()
    assert path.exists()
    assert (tmp_path / "detections.1.jsonl").exists()
