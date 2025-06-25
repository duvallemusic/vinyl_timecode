import os
import json
import librosa
from glob import glob
from tqdm import tqdm
import re

AUDIO_FOLDER = "vinyl_samples"
OUTPUT_JSON = "fingerprint_data.json"
SAMPLE_RATE = 22050
HOP_LENGTH = 512
WINDOW_DURATION = 0.1  # segundos

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
    return {
        "mfcc": mfcc.mean(axis=1).tolist(),
        "chroma": chroma.mean(axis=1).tolist(),
        "zcr": zcr.mean().item(),
        "spectral_centroid": spectral_centroid.mean().item()
    }

def generate_fingerprint_database():
    db = {}
    if not os.path.exists(AUDIO_FOLDER):
        print(f"[ERRO] Pasta '{AUDIO_FOLDER}' não encontrada.")
        return

    files = sorted(glob(os.path.join(AUDIO_FOLDER, "*.wav")))
    if not files:
        print("[ERRO] Nenhum arquivo .wav encontrado na pasta.")
        return

    pattern = re.compile(r"(?P<track>.+?)_(?P<variation>.+?)_\d+$")

    for file in tqdm(files, desc="Processando áudios"):
        base = os.path.splitext(os.path.basename(file))[0]
        match = pattern.match(base)
        if not match:
            print(f"[WARN] Formato de nome inesperado: {base}")
            track_name = base
            variation = "default"
        else:
            track_name = match.group("track")
            variation = match.group("variation")

        if track_name not in db:
            db[track_name] = {}
        if variation not in db[track_name]:
            db[track_name][variation] = {}
        y, sr = librosa.load(file, sr=SAMPLE_RATE)
        hop_samples = int(WINDOW_DURATION * sr)
        # Garante que só pega segmentos completos
        for start in range(0, len(y) - hop_samples + 1, hop_samples):
            end = start + hop_samples
            segment = y[start:end]
            timestamp = round(start / sr, 2)
            features = extract_features(segment, sr)
            db[track_name][variation][str(timestamp)] = features

    try:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(db, f, indent=2)
        print(f"[OK] Banco de fingerprints salvo em: {OUTPUT_JSON}")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar o arquivo JSON: {e}")

if __name__ == "__main__":
    generate_fingerprint_database()
