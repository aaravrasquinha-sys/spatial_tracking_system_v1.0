"""
Thin TensorRT engine runner implementing the PoseBackend protocol.

Section 3: "TensorRT engines are tied to the exact GPU, TensorRT version,
and JetPack build they were compiled on -- an engine built anywhere else
won't load." This module never builds an engine (see models/export_engine.py
for that); it only loads and runs one, on-device, batch size 1, FP16 I/O
as produced by the export step.

Imports of tensorrt/pycuda are deferred into __init__ so this module is
importable (and its pure-python siblings testable) on a machine without
either installed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from poi_perception.contracts import Frame
from poi_perception.inference import decode
from poi_perception.inference.pose_infer import RawDetection
from poi_perception.log import get_logger

log = get_logger("models.trt_engine")


class TRTPoseBackend:
    def __init__(
        self,
        engine_path: str,
        input_width: int = 640,
        input_height: int = 480,
        letterbox: bool = False,
        det_conf_thresh: float = 0.1,
        nms_iou_thresh: float = 0.45,
        keypoint_conf_thresh_for_decode: float = 0.05,
    ):
        try:
            import tensorrt as trt
            import pycuda.autoinit  # noqa: F401  -- initializes the CUDA context
            import pycuda.driver as cuda
        except ImportError as e:
            raise RuntimeError(
                "tensorrt/pycuda are not importable. This backend only runs "
                "on-device (the Orin, with JetPack's TensorRT installed). Use "
                "backend='ultralytics', or MockPoseBackend directly (see scripts/demo_synthetic.py), off-device."
            ) from e

        self._trt = trt
        self._cuda = cuda
        # Save a reference to the primary CUDA context for cross-thread push/pop support
        self._cuda_context = pycuda.autoinit.context

        self.input_width = input_width
        self.input_height = input_height
        self.letterbox = letterbox
        self.det_conf_thresh = det_conf_thresh
        self.nms_iou_thresh = nms_iou_thresh
        self.keypoint_conf_thresh_for_decode = keypoint_conf_thresh_for_decode

        self._check_manifest(engine_path)

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self._input_name = None
        self._output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_name = name
            else:
                self._output_name = name
        if self._input_name is None or self._output_name is None:
            raise RuntimeError(f"Could not resolve input/output tensor names in {engine_path}")

        self._alloc_buffers()
        log.info(f"Loaded TensorRT engine: {engine_path}")

    def _check_manifest(self, engine_path: str) -> None:
        """Section 3: lock and record JetPack/TensorRT version + export
        command next to the engine file. If the manifest is present,
        surface it in the log so a stale/foreign engine is obvious before
        it silently fails to load."""
        manifest_path = Path(engine_path).with_suffix(".manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            log.info(f"Engine manifest: {manifest}")
        else:
            log.warning(
                f"No manifest found next to {engine_path} (expected "
                f"{manifest_path.name}). Section 3 requires recording the "
                f"JetPack version, TensorRT version, and export command next "
                f"to the engine -- re-export with models/export_engine.py."
            )

    def _alloc_buffers(self) -> None:
        trt, cuda = self._trt, self._cuda
        in_shape = (1, 3, self.input_height, self.input_width)
        self.context.set_input_shape(self._input_name, in_shape)
        out_shape = tuple(self.context.get_tensor_shape(self._output_name))

        self._in_host = np.empty(in_shape, dtype=np.float32)
        self._out_host = np.empty(out_shape, dtype=np.float32)
        self._in_device = cuda.mem_alloc(self._in_host.nbytes)
        self._out_device = cuda.mem_alloc(self._out_host.nbytes)
        self._stream = cuda.Stream()

        self.context.set_tensor_address(self._input_name, int(self._in_device))
        self.context.set_tensor_address(self._output_name, int(self._out_device))

    def warm_up(self, n: int = 5) -> None:
        dummy = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        for _ in range(n):
            self._infer_raw(dummy)

    def _infer_raw(self, rgb: np.ndarray) -> tuple:
        cuda = self._cuda
        
        # Push the CUDA context onto the current thread's stack to allow 
        # multi-threaded daemon execution (e.g., inference running in a worker thread)
        self._cuda_context.push()
        try:
            chw, letterbox_params = decode.preprocess(rgb, self.input_width, self.input_height, self.letterbox)
            np.copyto(self._in_host, chw)

            cuda.memcpy_htod_async(self._in_device, self._in_host, self._stream)
            self.context.execute_async_v3(stream_handle=self._stream.handle)
            cuda.memcpy_dtoh_async(self._out_host, self._out_device, self._stream)
            self._stream.synchronize()

            return self._out_host, letterbox_params
        finally:
            cuda.Context.pop()

    def infer(self, frame: Frame) -> List[RawDetection]:
        src_h, src_w = frame.rgb.shape[:2]
        raw_output, letterbox_params = self._infer_raw(frame.rgb)
        return decode.decode_yolo_pose_output(
            raw_output,
            letterbox_params,
            src_w,
            src_h,
            self.det_conf_thresh,
            self.nms_iou_thresh,
            self.keypoint_conf_thresh_for_decode,
        )