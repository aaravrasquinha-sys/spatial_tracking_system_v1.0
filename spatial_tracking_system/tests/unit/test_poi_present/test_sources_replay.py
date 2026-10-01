import asyncio
import json

import pytest

pytest.importorskip("poi_localization")

from poi_present.config import SourceConfig
from poi_present.sources.replay import ReplaySource


def _write_log(path, phase="A"):
    from poi_localization.io.world_track import JsonlWorldTrackWriter, WorldTrackPhaseA

    w = JsonlWorldTrackWriter(path, phase=phase)
    for i in range(3):
        rec = WorldTrackPhaseA(
            t_capture=100.0 + i * 0.1,
            world_track_id=1,
            state="confirmed",
            p_local=(float(i), 0.0, 0.0),
            v_local=(1.0, 0.0, 0.0),
            cov_xy=(0.01, 0.0, 0.01),
            height_m=1.7,
            src="fused",
            age_s=float(i),
        )
        w.write_frame(rec.t_capture, i, [rec])
    w.close()


@pytest.mark.asyncio
async def test_replay_source_no_loop_terminates(tmp_path):
    log_path = tmp_path / "cam0_test.jsonl"
    _write_log(log_path)

    cfg = SourceConfig(kind="replay", replay_path=str(log_path), replay_speed=1000.0, replay_loop=False)
    source = ReplaySource(cfg)

    bundles = []

    async def sink(bundle):
        bundles.append(bundle)

    await asyncio.wait_for(source.run(sink), timeout=5.0)

    assert len(bundles) == 3
    assert bundles[0].frame == "local"
    assert bundles[0].tracks[0].p == (0.0, 0.0, 0.0)
    assert bundles[2].tracks[0].p == (2.0, 0.0, 0.0)


@pytest.mark.asyncio
async def test_replay_source_loops_until_stopped(tmp_path):
    log_path = tmp_path / "cam0_test.jsonl"
    _write_log(log_path)

    cfg = SourceConfig(kind="replay", replay_path=str(log_path), replay_speed=1000.0, replay_loop=True)
    source = ReplaySource(cfg)

    bundles = []

    async def sink(bundle):
        bundles.append(bundle)
        if len(bundles) >= 7:  # more than one loop's worth of 3 frames
            source.stop()

    await asyncio.wait_for(source.run(sink), timeout=5.0)
    assert len(bundles) >= 7


@pytest.mark.asyncio
async def test_replay_source_phase_b_maps_to_room_frame(tmp_path):
    from poi_localization.io.world_track import JsonlWorldTrackWriter, WorldTrackPhaseB

    log_path = tmp_path / "cam0_b.jsonl"
    w = JsonlWorldTrackWriter(log_path, phase="B", map_id="abc123")
    rec = WorldTrackPhaseB(
        id=1, state="confirmed", p=(1.0, 1.0, 0.0), v=(0.0, 0.0, 0.0),
        cov_xy=(0.01, 0.0, 0.01), height=1.7, conf=0.9, src="fused",
        age_s=1.0, bbox=(0, 0, 10, 10),
    )
    w.write_frame(100.0, 0, [rec])
    w.close()

    cfg = SourceConfig(kind="replay", replay_path=str(log_path), replay_speed=1000.0, replay_loop=False)
    source = ReplaySource(cfg)
    bundles = []

    async def sink(bundle):
        bundles.append(bundle)

    await asyncio.wait_for(source.run(sink), timeout=5.0)
    assert bundles[0].frame == "room"
    assert bundles[0].map_id == "abc123"
    assert bundles[0].tracks[0].conf == 0.9
