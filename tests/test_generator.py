import os
import sys
import json
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import generator


def create_tone(path, sr=generator.SAMPLE_RATE, duration=None, freq=440.0):
    if duration is None:
        duration = generator.WINDOW_DURATION * generator.NUM_WINDOWS
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(path, data, sr)


def test_generate_fingerprint_database(tmp_path, monkeypatch):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    tone_path = sample_dir / "tone_33rpm_0.wav"
    create_tone(tone_path)

    output_json = tmp_path / "fingerprint.json"

    monkeypatch.setattr(generator, "AUDIO_FOLDER", str(sample_dir))
    monkeypatch.setattr(generator, "OUTPUT_JSON", str(output_json))

    generator.generate_fingerprint_database()

    assert output_json.exists()
    with open(output_json) as f:
        data = json.load(f)
    assert "tone" in data
