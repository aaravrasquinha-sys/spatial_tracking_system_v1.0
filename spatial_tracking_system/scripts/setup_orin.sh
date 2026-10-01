#!/usr/bin/env bash
# WP-J0: Jetson Orin Nano (8GB, JetPack 6.2 / L4T R36.4.x) bring-up.
#
# Run this ONCE on the target machine, in stages -- it's deliberately
# split into functions you can call individually (see "STAGES" below)
# rather than one blind end-to-end script, because a few steps
# (librealsense udev rules, a reboot after jetson_clocks/nvpmodel) need
# you in the loop, and because GTSAM's build is long enough (~30-60min
# on a 6-core Orin Nano) that you don't want a failure in a LATER stage
# to force redoing it.
#
# Usage:
#   chmod +x scripts/setup_orin.sh
#   ./scripts/setup_orin.sh all          # everything, in order
#   ./scripts/setup_orin.sh deps         # apt + python deps only
#   ./scripts/setup_orin.sh librealsense # librealsense w/ CUDA + udev rules
#   ./scripts/setup_orin.sh gtsam        # GTSAM from source, built against THIS numpy
#   ./scripts/setup_orin.sh probe        # env_probe.py --realsense (run after the above)
#   ./scripts/setup_orin.sh power        # WP-K4: show this unit's power modes + which one to use
#   ./scripts/setup_orin.sh power apply  # ...and actually switch to it + lock clocks (sudo)
#
# After this script: `python3 -m pyslam.selftest`, then
# `python3 -m pyslam.tools.env_probe --realsense`, per the Runbook.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${HOME}/orin_build"
JOBS="$(nproc)"

log() { echo -e "\n\033[1;36m[setup_orin]\033[0m $*"; }
warn() { echo -e "\033[1;33m[setup_orin][WARN]\033[0m $*"; }

check_jetson() {
    if [ ! -f /proc/device-tree/model ] || ! grep -qi jetson /proc/device-tree/model; then
        warn "This does not look like a Jetson (/proc/device-tree/model missing or doesn't mention 'jetson')."
        warn "Continuing anyway, but review each step before trusting it."
    else
        log "Detected: $(tr -d '\0' < /proc/device-tree/model)"
    fi
    if [ -f /etc/nv_tegra_release ]; then
        log "L4T: $(cat /etc/nv_tegra_release)"
    fi
}

stage_deps() {
    log "Installing apt + Python dependencies (does NOT touch numpy's version -- "
    log "see stage_gtsam for why that matters)."
    sudo apt-get update
    sudo apt-get install -y \
        build-essential cmake git pkg-config \
        libboost-all-dev libtbb-dev \
        python3-dev python3-pip python3-venv \
        libgtk-3-dev libcanberra-gtk3-module \
        libssl-dev libusb-1.0-0-dev libudev-dev \
        libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
        wget curl

    # Project's own Python deps. numpy is left to whatever this
    # environment already resolves (>=2.0, matching every gate/module
    # this codebase already ships) -- GTSAM below is built FROM SOURCE
    # specifically so it links against this numpy, not a `pip install
    # gtsam` wheel that pulls numpy<2 and would silently downgrade
    # everything else (see WP_P4_Findings.md, "iSAM2 NOT attempted" --
    # this stage is what closes that gap).
    pip3 install --upgrade pip
    pip3 install numpy scipy scikit-learn opencv-contrib-python
    log "deps stage done. Verify: python3 -c 'import numpy; print(numpy.__version__)' should print >=2.0"
}

stage_librealsense() {
    log "Building librealsense with CUDA-accelerated align/pointcloud."
    log "This is what makes rs.align() (used by pyslam/sensors/realsense.py) run on the GPU"
    log "instead of the CPU -- accuracy-neutral (same output), pure speed."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    if [ ! -d librealsense ]; then
        git clone --depth 1 --branch master https://github.com/IntelRealSense/librealsense.git
    fi
    cd librealsense

    # udev rules -- required for non-root USB access to the D435i.
    sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules && sudo udevadm trigger

    mkdir -p build && cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_WITH_CUDA=ON \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DPYTHON_EXECUTABLE="$(which python3)" \
        -DFORCE_RSUSB_BACKEND=ON \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_GRAPHICAL_EXAMPLES=OFF
    # FORCE_RSUSB_BACKEND=ON: Jetson's kernel is NOT the patched one
    # Intel ships for x86 (that patch touches the mainline uvcvideo
    # driver); RSUSB is librealsense's own userspace USB backend and is
    # the documented, supported path on Jetson. env_probe.py's
    # "usb_type" field is worth checking against this after install --
    # if it reports something unexpected, the RSUSB backend is why.
    make -j"${JOBS}"
    sudo make install
    sudo ldconfig

    # Python bindings land in the build tree; symlink pyrealsense2 into
    # this interpreter's site-packages so `import pyrealsense2` works
    # the same way it does on the reference x86 machine.
    PY_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
    PYRS_BUILD="$(find "${BUILD_DIR}/librealsense/build" -maxdepth 3 -name 'pyrealsense2*.so' | head -1)"
    if [ -n "${PYRS_BUILD}" ]; then
        cp "${PYRS_BUILD}" "${PY_SITE}/"
        log "Installed pyrealsense2 to ${PY_SITE}"
    else
        warn "Could not locate the built pyrealsense2*.so -- check the build/ tree manually "
        warn "under ${BUILD_DIR}/librealsense/build/ and copy it into your interpreter's "
        warn "site-packages, or run 'sudo make install' output above for its actual path."
    fi
    log "librealsense stage done. Verify: python3 -c 'import pyrealsense2; print(pyrealsense2.__version__)'"
    log "Then: python3 -m pyslam.tools.env_probe --realsense   (with the camera plugged in)"
}

stage_gtsam() {
    log "Building GTSAM from source against THIS interpreter's numpy (>=2.0)."
    log "'pip install gtsam' pulls a numpy<2 pin (see WP_P4_Findings.md) -- building from"
    log "source with -DGTSAM_PYTHON_VERSION pinned to this interpreter avoids that entirely"
    log "and is what makes iSAM2 (Tier 2 of the port plan) reachable on this machine."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    if [ ! -d gtsam ]; then
        git clone --depth 1 --branch release/4.2 https://github.com/borglab/gtsam.git
    fi
    cd gtsam
    mkdir -p build && cd build
    PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGTSAM_BUILD_PYTHON=ON \
        -DGTSAM_PYTHON_VERSION="${PY_VER}" \
        -DGTSAM_USE_SYSTEM_EIGEN=OFF \
        -DGTSAM_BUILD_TESTS=OFF \
        -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
        -DGTSAM_WITH_TBB=ON \
        -DGTSAM_BUILD_UNSTABLE=ON
    make -j"${JOBS}" python-install
    log "gtsam stage done. Verify: python3 -c 'import gtsam; print(gtsam.__version__)'"
    log "Then: python3 -c 'import numpy; print(numpy.__version__)' should STILL be >=2.0"
    log "(if it silently dropped below 2.0, something in this build pulled a wheel -- check"
    log "'pip show gtsam' points at this build dir, not PyPI)."
}

stage_power() {
    # WP-K4 (Phase A4). The old README said `sudo nvpmodel -m 0` = "MAXN".
    # On the Orin Nano dev kit under JetPack 6.2+ that is WRONG: mode 0 is
    # the 15W mode (confirmed on the target unit: `nvpmodel -q` printed
    # "15W / 0" and tegrastats showed CPU 1497MHz, GPU 611MHz) -- so every
    # timing measured after following the old instructions was taken at
    # power-limited clocks. Mode IDs are NOT stable across Jetson models or
    # JetPack releases, so this looks the max-performance mode up BY NAME
    # in this unit's own config instead of hard-coding an ID.
    local action="${1:-show}"
    local conf="${NVPMODEL_CONF:-/etc/nvpmodel.conf}"

    if command -v nvpmodel >/dev/null 2>&1; then
        log "Current power mode:"
        sudo nvpmodel -q || warn "nvpmodel -q failed"
    else
        warn "nvpmodel not found on PATH (not a Jetson?)"
    fi

    if [ ! -f "${conf}" ]; then
        warn "${conf} not found; run 'sudo nvpmodel -p --verbose' and choose the top mode by hand."
        return 0
    fi

    log "Power modes defined on THIS unit (${conf}):"
    local modes
    modes="$(grep -E '^[[:space:]]*<[[:space:]]*POWER_MODEL[[:space:]]+ID=' "${conf}" \
             | sed -E 's/.*ID=([0-9]+)[[:space:]]+NAME=([^ >]+).*/\1 \2/' || true)"
    if [ -z "${modes}" ]; then
        warn "no POWER_MODEL lines parsed from ${conf}; run 'sudo nvpmodel -p --verbose'."
        return 0
    fi
    echo "${modes}" | sed 's/^/    mode /'

    # Prefer a *SUPER* MAXN variant, else any MAXN*.
    local pick
    pick="$(echo "${modes}" | grep -i 'MAXN' | grep -i 'SUPER' | head -1 || true)"
    if [ -z "${pick}" ]; then
        pick="$(echo "${modes}" | grep -i 'MAXN' | head -1 || true)"
    fi
    if [ -z "${pick}" ]; then
        warn "no MAXN* mode listed above -- choose the highest-performance one by name yourself."
        return 0
    fi
    local id name
    id="${pick%% *}"; name="${pick#* }"
    log "Recommended for benchmarking and live runs: mode ${id} (${name})"

    if [ "${action}" != "apply" ]; then
        echo
        echo "    sudo nvpmodel -m ${id}   # ${name}"
        echo "    sudo jetson_clocks       # lock clocks (no DVFS ramp lag)"
        echo
        log "Nothing changed. Re-run './scripts/setup_orin.sh power apply' to do exactly the two commands above."
        warn "Max clocks raise power draw and heat; check tegrastats (tj temp) during a long run."
        return 0
    fi

    sudo nvpmodel -m "${id}"
    sudo jetson_clocks
    log "Now:"
    sudo nvpmodel -q || true
    sudo jetson_clocks --show | head -12 || true
    warn "If nvpmodel asked for a reboot, reboot and re-run 'power apply'. Confirm with tegrastats: CPU and GPU MHz should be well above the 15W-mode values (CPU 1497 / GPU 611)."
}

stage_probe() {
    cd "${REPO_ROOT}"
    log "Running env_probe.py. Plug in the D435i first if you want --realsense to mean anything."
    python3 -m pyslam.tools.env_probe --realsense || true
    log "Review the 'jetson' block above: RAM, JetPack/L4T, CUDA version, OpenCV CUDA status,"
    log "nvpmodel/jetson_clocks state. Set MAXN + jetson_clocks before any timing comparison:"
    echo
    echo "    ./scripts/setup_orin.sh power          # list THIS unit's power modes + the one to use"
    echo "    ./scripts/setup_orin.sh power apply    # switch to it + jetson_clocks (WP-K4: do NOT assume mode 0 = MAXN)"
    echo
    warn "jetson_clocks holds clocks at max continuously -- fine for a benchmarking session,"
    warn "but increases idle power/thermal draw. Don't leave it on unattended long-term "
    warn "without checking your enclosure's cooling."
}

case "${1:-}" in
    deps) check_jetson; stage_deps ;;
    librealsense) check_jetson; stage_librealsense ;;
    gtsam) check_jetson; stage_gtsam ;;
    probe) stage_probe ;;
    power) check_jetson; stage_power "${2:-show}" ;;
    all)
        check_jetson
        stage_deps
        stage_power show   # WP-K4: shows, never applies -- max clocks are a deliberate choice
        stage_librealsense
        stage_gtsam
        stage_probe
        ;;
    *)
        echo "Usage: $0 {deps|librealsense|gtsam|probe|power [apply]|all}"
        exit 1
        ;;
esac
