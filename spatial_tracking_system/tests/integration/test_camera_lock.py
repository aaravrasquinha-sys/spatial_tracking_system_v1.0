import multiprocessing as mp
import time

import pytest

from sts.camera_lock import CameraBusy, CameraLock


def test_second_owner_is_refused_with_the_holders_name(tmp_path):
    with CameraLock(tmp_path, "d", "map"):
        with pytest.raises(CameraBusy, match="sts map"):
            CameraLock(tmp_path, "d", "run").acquire()
    CameraLock(tmp_path, "d", "run").acquire().release()


def test_different_cameras_do_not_block_each_other(tmp_path):
    with CameraLock(tmp_path, "cam0", "run"):
        CameraLock(tmp_path, "cam1", "run").acquire().release()


def _hold(path, ready):
    CameraLock(path, "d", "child").acquire()
    ready.set()
    time.sleep(30)


def test_a_crashed_holder_does_not_leave_a_stale_lock(tmp_path):
    ctx = mp.get_context("fork")
    ready = ctx.Event()
    p = ctx.Process(target=_hold, args=(str(tmp_path), ready)); p.start()
    assert ready.wait(10)
    with pytest.raises(CameraBusy):
        CameraLock(tmp_path, "d", "x").acquire()
    p.kill(); p.join()
    CameraLock(tmp_path, "d", "x").acquire().release()       # kernel released the flock with the process
