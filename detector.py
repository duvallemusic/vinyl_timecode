import json
import librosa
from scipy.spatial.distance import cosine

SAMPLE_RATE = 22050
HOP_LENGTH = 512
FINGERPRINT_FILE = "fingerprint_data.json"

class FingerprintDetector:
    def __init__(self):
        with open(FINGERPRINT_FILE, "r") as f:
            self.fingerprint_db = json.load(f)

    def extract_features(self, y):
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13, hop_length=HOP_LENGTH)
        chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        return {
            "mfcc": mfcc.mean(axis=1).tolist(),
            "chroma": chroma.mean(axis=1).tolist(),
            "zcr": zcr.mean().item(),
            "spectral_centroid": spectral_centroid.mean().item()
        }

    def compare_features(self, live, stored_list):
        distances = []
        for f in stored_list:
            d = (
                0.5 * cosine(live["mfcc"], f["mfcc"]) +
                0.3 * cosine(live["chroma"], f["chroma"]) +
                0.1 * abs(live["zcr"] - f["zcr"]) +
                0.1 * abs(live["spectral_centroid"] - f["spectral_centroid"])
            )
            distances.append(d)
        return min(distances)

    def get_live_position(self, audio_buffer, track_name="g-funk-demo", variation="33rpm"):
        features = self.extract_features(audio_buffer)
        best_timestamp = None
        best_distance = float("inf")

        track_data = self.fingerprint_db.get(track_name, {}).get(variation, {})
        for timestamp_str, feature_list in track_data.items():
            dist = self.compare_features(features, [feature_list])
            if dist < best_distance:
                best_distance = dist
                best_timestamp = float(timestamp_str)

        return best_timestamp, best_distance