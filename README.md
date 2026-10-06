# StepTracker

Tracks procedure steps from a fixed camera by matching live frames to reference images
of the unit after each step (DINOv2 patch features, no training needed).

## Setup
Venv is at `~/venvs/steptracker` (torch, transformers, opencv).

## Reference layout
One folder per state, named so they sort in procedure order. Put images and/or short videos in each:
```
refs/
  00_start/            unit before any step
  01_remove_cover/     unit after step 1
  02_remove_screws/    unit after step 2
```
Keep the camera, lighting and framing identical to the live setup. Include a few
variations per state (hands in view, slight lighting changes).

## GUI
```
~/venvs/steptracker/bin/python gui.py [--refs refs] [--source 0]
```
Start/stop on a camera index or video file, drag a crop region on the video, add states,
capture reference frames from the live feed (or import files), and tune thresholds live.
Reference changes reload while running. Needs `PySide6-Essentials` (already in the venv).

## Run (CLI)
```
~/venvs/steptracker/bin/python track.py --refs refs --source 0             # webcam
~/venvs/steptracker/bin/python track.py --refs refs --source video.mp4     # recording
~/venvs/steptracker/bin/python track.py --refs refs --source 0 --roi       # crop to the unit first
```
Prints `STEP N COMPLETE`, and warns when a step is skipped or the unit goes back to an earlier state.

## Tuning
- `--confirm` frames a state must be matched before a step counts (default 8, raise to cut false alarms)
- `--margin` required similarity lead over the second-best state (default 0.02)
- `--window` smoothing frames, `--every` process every Nth frame
