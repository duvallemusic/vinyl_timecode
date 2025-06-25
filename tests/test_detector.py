import os
import sys
import json
import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import detector


def create_tone(path, sr=detector.SAMPLE_RATE, duration=None, freq=440.0):
    if duration is None:
        duration = detector.WINDOW_DURATION * detector.NUM_WINDOWS
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(path, data, sr)
    return data


def test_get_live_position(tmp_path, monkeypatch):
    tone_path = tmp_path / "tone.wav"
    audio = create_tone(tone_path)

    temp_db = tmp_path / "db.json"
    features = detector.FingerprintDetector.extract_features(detector.FingerprintDetector.__new__(detector.FingerprintDetector), audio)
    db = {"tone": {"33rpm": {"0.0": features}}}
    with open(temp_db, "w") as f:
        json.dump(db, f)

    monkeypatch.setattr(detector, "FINGERPRINT_FILE", str(temp_db))
    fd = detector.FingerprintDetector()
    position, distance = fd.get_live_position(audio, track_name="tone")

    assert position == 0.0
    assert distance == 0.0
