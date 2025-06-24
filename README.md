# Vinyl Timecode

This project demonstrates vinyl fingerprint generation and detection. It analyzes small audio windows from vinyl recordings and stores them in a JSON database. The detector can then estimate the playback position of a record and synchronize a digital track.

## Setup

1. **Install requirements**

   Install Python dependencies using pip:
   ```bash
   pip install librosa sounddevice soundfile numpy scipy tqdm
   ```

2. **Generate the fingerprint database**

   Run `main.py` to process WAV files inside the `vinyl_samples` directory:
   ```bash
   python main.py
   ```
   This produces `fingerprint_data.json` with features for each timestamp.

3. **Run synchronized playback**

   Ensure `track_22050.wav` exists (convert `track.wav` with `convert_to_22050.py` if needed) and execute:
   ```bash
   python sync_playback.py
   ```
   The script listens to the vinyl audio and plays the digital track in sync.

## Files

- `vinyl_samples/*.wav` – sample vinyl recordings used for fingerprint creation.
- `track.wav` – original digital track.
- `track_22050.wav` – resampled version used during playback.
- `fingerprint_data.json` – generated fingerprint database.

Running the above steps will create the database and start synchronized playback between the vinyl and the digital track.
