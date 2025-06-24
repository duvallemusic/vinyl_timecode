# Vinyl Timecode

This repository contains a collection of scripts for generating audio fingerprints and
synchronising playback with a physical vinyl record.

## Requirements

Install the Python dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The main packages used are `numpy`, `scipy`, `librosa`, `soundfile`, `sounddevice`
and `tqdm`.

## Scripts

- `generator.py` – builds a JSON database of fingerprints from sample audio files.
- `player.py` – plays a track and accepts simple console commands.
- `convert_to_22050.py` – converts an audio file to 22.05 kHz.
- `sync_playback.py` – attempts to synchronise playback with a live vinyl source.
- `detector.py` – utilities for comparing audio fingerprints.

Samples and example tracks are located in the `vinyl_samples` folder.
