import json
import librosa
import numpy as np
from scipy.spatial.distance import cosine

SAMPLE_RATE = 22050
HOP_LENGTH = 512
WINDOW_DURATION = 0.1
NUM_WINDOWS = 3
FINGERPRINT_FILE = "fingerprint_data.json"

class FingerprintDetector:
    def __init__(self):
        with open(FINGERPRINT_FILE, "r") as f:
            self.fingerprint_db = json.load(f)

    def extract_features(self, y):
        """Retorna um vetor do espectrograma mel."""
        S = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_mels=20, n_fft=1024, hop_length=HOP_LENGTH
        )
        log_S = librosa.power_to_db(S, ref=np.max)
        return log_S.flatten().tolist()

    def compare_features(self, live, stored_list):
        distances = []
        live_arr = np.array(live)
        for f in stored_list:
            try:
                f_arr = np.array(f)
                if len(f_arr) != len(live_arr):
                    distances.append(float("inf"))
                else:
                    distances.append(cosine(live_arr, f_arr))
            except Exception:
                distances.append(float("inf"))
        return min(distances)

    def get_live_position(self, audio_buffer, track_name="g-funk-demo", variation="33rpm"):
        """Retorna (timestamp, distância) da melhor correspondência."""
        features = self.extract_features(audio_buffer)
        best_timestamp = None
        best_distance = float("inf")

        track_data = self.fingerprint_db.get(track_name, {}).get(variation, {})
        for timestamp_str, stored_features in track_data.items():
            dist = self.compare_features(features, [stored_features])
            if dist < best_distance:
                best_distance = dist
                best_timestamp = float(timestamp_str)

        return best_timestamp, best_distance
