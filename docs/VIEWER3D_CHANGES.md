# STS 3D Viewer – Step-by-step Change Guide

## Overview and prep

You will add a desktop 3D viewer at `http://192.168.1.101:8765/viewer3d/` with 2 small edits, 3 new files and one downloaded library. Part B (optional, last section) adds real 3D skeleton joints and touches the tracking code, so do Steps 1 to 7 first and check them live before attempting it.

All paths are relative to `/home/drdo/spatial_tracking_system_v1.0` on the Jetson. Tested here against the synthetic source with a fake room scan; not yet run on your RealSense.

| Step | File | Action |
| --- | --- | --- |
| 1 | `scripts/run_present.py` | add 1 line |
| 2 | `web/common/vendor/three/jsm/controls/OrbitControls.js` | download |
| 3 | `web/viewer3d/index.html` | new file |
| 4 | `web/viewer3d/style.css` | new file |
| 5 | `web/viewer3d/app.js` | new file |
| 6 | `web/index.html` | add 1 link |

Before you start, back up and branch so you can undo everything:

```bash
cd /home/drdo/spatial_tracking_system_v1.0
git status            # if this is a git checkout
git checkout -b viewer3d
# no git? then: cp -r web web.bak && cp scripts/run_present.py scripts/run_present.py.bak
```

The viewer needs no change to the running backend except Step 1. Restart `run_present.py` after Step 1 so it takes effect.

## Step 1: fix the map ID in `scripts/run_present.py`

One line fixes a real bug: `LiveSource` is created with `map_id=None`, so on the first frame the server overwrites the scene's real map ID (`7814e6723421`) with null, and no viewer can show which map it is registered to.

Find this block inside `_build_live` (about line 78):

```python
    T = m4_pipeline.floor_transform
    server.scene.update_camera(T.R.tolist(), T.t.tolist(), T.camera_height_m, frame_name, T.map_id)
```

Insert one line between the two so it reads:

```python
    T = m4_pipeline.floor_transform
    live_source.map_id = T.map_id  # was left as None, which overwrote the scene's real map_id on the first frame
    server.scene.update_camera(T.R.tolist(), T.t.tolist(), T.camera_height_m, frame_name, T.map_id)
```

Nothing else changes. The existing dashboard keeps working.

## Step 2: add OrbitControls (the mouse rotate/pan/zoom library)

Your repo already vendors Three.js r160 and `PLYLoader`, but not `OrbitControls`. It must be the same r160 release. On the Jetson (needs internet once):

```bash
cd /home/drdo/spatial_tracking_system_v1.0
mkdir -p web/common/vendor/three/jsm/controls
curl -fsSL https://raw.githubusercontent.com/mrdoob/three.js/r160/examples/jsm/controls/OrbitControls.js \
  -o web/common/vendor/three/jsm/controls/OrbitControls.js
wc -c web/common/vendor/three/jsm/controls/OrbitControls.js   # expect about 29868 bytes
head -12 web/common/vendor/three/jsm/controls/OrbitControls.js | grep -c "from 'three'"   # expect 1
```

No internet on the Jetson? Run the same `curl` on your laptop and `scp` the file to that path. The viewer does not load anything from a CDN at runtime, so it still works on an offline LAN afterwards.

## Steps 3 to 5: create the three viewer files

Create the folder and the three files below. Paste each block exactly; the file name is in the heading above it.

```bash
mkdir -p /home/drdo/spatial_tracking_system_v1.0/web/viewer3d
```

### Step 3: `web/viewer3d/index.html`

[HTML content - refer to the original doc for the complete code]

### Step 4: `web/viewer3d/style.css`

[CSS content - refer to the original doc for the complete code]

### Step 5: `web/viewer3d/app.js`

The file is 925 lines. It consists of two parts that must be pasted together in order.

[JavaScript content - refer to the original doc for the complete code]

Check that the two parts joined correctly: `node --check web/viewer3d/app.js` should print nothing (if Node is installed), and the file should be about 925 lines (`wc -l`). If the browser shows a blank page, open the developer console (F12): a 404 on `OrbitControls.js` means Step 2 was skipped or went to the wrong path.

## Step 6: link the page from `web/index.html`

Optional but handy. In `web/index.html`, find the VR link line:

```html
    <a class="link" href="vr/"><span>WebXR viewer</span><span class="go mono">VR →</span></a>
```

and add this line directly above it:

```html
    <a class="link" href="viewer3d/"><span>3D viewer</span><span class="go mono">3D →</span></a>
```

The picker at `http://192.168.1.101:8765/` then lists Dashboard, 3D viewer and WebXR viewer.

## Step 7: run it and check it

Stop the running `run_present.py` (Ctrl+C) and start it again with your usual command, so Step 1 takes effect:

```bash
cd /home/drdo/spatial_tracking_system_v1.0
python3 scripts/run_present.py \
  --source live \
  --m3-config data/runtime/site_001/cam0/m3.json \
  --m4-config data/runtime/site_001/cam0/m4.json \
  --present-config data/runtime/site_001/cam0/present.json \
  --m4-source realsense
```

Open `http://192.168.1.101:8765/viewer3d/` from your desktop browser. Keep the trailing slash: without it the page's relative file paths break.

**What you should see**

| Check | Expected |
| --- | --- |
| Top bar | green dot and `live`, a green `room` badge, the map badge `7814e6723421`, fps from the Jetson |
| Banner | none. A yellow `LOCAL FRAME` or red `CALIBRATION …` banner means positions are not trustworthy |
| Room | the scanned room appears upright (floor on the grid, ceiling above). A first load can download up to about 22 MB; a progress message shows |
| Left panel, map line | a point count and `z 0.0 … ~2.6 m`. `CHECK FRAME` appended means the floor is not near z = 0 |
| Ceiling cut | starts about 0.3 m below the top of the scan. Drag it right to `off`, left to see into the room from above |
| People | a box on the floor point with an ID label, a ground ellipse for uncertainty and a 5 s trail. Dashed outline = coasting or predicted |
| Mouse | drag orbits, right-drag pans, wheel zooms, double-click a person follows them, `Esc` releases |
| Keys | `1` top, `2` iso, `3` sensor view, `T` trails, `L` labels |

Walk to a spot you can measure (for example 2 m from a wall) and check that the box stands there in the 3D room. This is the real test of the calibration; the viewer can only show what the tracker reports.

**If something is wrong**

| Symptom | Likely cause and fix |
| --- | --- |
| Blank page, console shows 404 for `OrbitControls.js` | Step 2 file missing or in the wrong folder |
| Blank page, other 404s | missing trailing slash, or a file name typo in `web/viewer3d/` |
| 401 Unauthorized | `present.token` is set; open `…/viewer3d/?token=YOURTOKEN` |
| No room, grid only | run `curl -s http://192.168.1.101:8765/api/scene \| python3 -m json.tool \| head -30`; `assets` should list `/map/viewer/points.ply`. If empty, `present.json` has no `map_bundle_dir` |
| Old room after re-calibrating | the server caches `/map/` files for a day; hard refresh with Ctrl+Shift+R |
| Room lies on its side | the file is not in the room frame (`CHECK FRAME` shown); tell me the console bbox line |
| Walls hide everyone | lower the ceiling cut, switch Person boxes off to see only trails, or use the top view |
| `stream stalled` | the tracker is publishing nothing (nobody in view still sends empty frames); check the `run_present.py` terminal |
| Jittery boxes | normal at low fps from the camera; the viewer renders 0.1 s behind live to smooth them |

The person boxes are derived from the tracker's floor point and estimated height, drawn about 0.55 m by 0.40 m. They are not measured 3D bounding boxes; the stream does not contain any. Real skeletons need Part B.

---

**Part B is optional.** It adds 3D skeleton lifting from the depth image. Do Steps 1–7 first and confirm the viewer works. See the original document for full Part B details.

