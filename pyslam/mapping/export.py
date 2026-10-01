import numpy as np


def write_ply(path: str, pts: np.ndarray, colors: np.ndarray) -> None:
    n = pts.shape[0]
    with open(path, "wb") as f:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        f.write(header)
        if n == 0:
            return
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                           ("r", "u1"), ("g", "u1"), ("b", "u1")])
        buf = np.zeros(n, dtype=dtype)
        buf["x"], buf["y"], buf["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        buf["r"], buf["g"], buf["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
        f.write(buf.tobytes())
