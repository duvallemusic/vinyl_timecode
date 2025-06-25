import os
import json
import librosa
import numpy as np
import sounddevice as sd
from glob import glob
from tqdm import tqdm

# Constantes
AUDIO_FOLDER = "vinyl_samples"
OUTPUT_JSON = "fingerprint_data.json"
SAMPLE_RATE = 22050
HOP_LENGTH = 512
WINDOW_DURATION = 0.1  # segundos
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)

# Banco de dados de fingerprint
fingerprint_db = {}

def generate_fingerprint_database():
    """
    Gera fingerprints por janela e salva em JSON.
    """
    print(f"Procurando arquivos em: {AUDIO_FOLDER}")
    for variation, count in VARIATIONS.items():
        pattern = os.path.join(AUDIO_FOLDER, f"*_{variation}_*.wav")
        files = sorted(glob(pattern))[:count]
        for file in tqdm(files):
            track_name = os.path.basename(file).split("_")[0]
            if track_name not in fingerprint_db:
                fingerprint_db[track_name] = {}
            fingerprint_db[track_name][variation] = []

            y, sr = librosa.load(file, sr=SAMPLE_RATE)
            windows = extract_window_features(y, sr)
            fingerprint_db[track_name][variation] = windows

    with open(OUTPUT_JSON, "w") as f:
        json.dump(fingerprint_db, f, indent=4)
    print(f"Fingerprint salvo em: {OUTPUT_JSON}")

if not os.path.exists(OUTPUT_JSON):
    generate_fingerprint_database()

# Importar depois que o JSON foi gerado
from detector import FingerprintDetector

VARIATIONS = {
    "33rpm": 2,  # apenas 33rpm será usada no detector
}

# Banco de dados de fingerprint
fingerprint_db = {}

# -------------------
# Funções utilitárias
# -------------------

def extract_window_features(audio, sr):
    """
    Divide o áudio em janelas e extrai os features de cada uma.
    """
    features_per_window = []
    total_samples = len(audio)
    for start in range(0, total_samples - WINDOW_SIZE, WINDOW_SIZE):
        end = start + WINDOW_SIZE
        window = audio[start:end]
        features = detector.extract_features(window)
        features_per_window.append(features)
    return features_per_window

def generate_fingerprint_database():
    """
    Gera fingerprints por janela e salva em JSON.
    """
    print(f"Procurando arquivos em: {AUDIO_FOLDER}")
    for variation, count in VARIATIONS.items():
        pattern = os.path.join(AUDIO_FOLDER, f"*_{variation}_*.wav")
        files = sorted(glob(pattern))[:count]
        for file in tqdm(files):
            track_name = os.path.basename(file).split("_")[0]
            if track_name not in fingerprint_db:
                fingerprint_db[track_name] = {}
            fingerprint_db[track_name][variation] = []

            y, sr = librosa.load(file, sr=SAMPLE_RATE)
            windows = extract_window_features(y, sr)
            fingerprint_db[track_name][variation] = windows

    with open(OUTPUT_JSON, "w") as f:
        json.dump(fingerprint_db, f, indent=4)
    print(f"Fingerprint salvo em: {OUTPUT_JSON}")

def load_fingerprint_database():
    """
    Carrega os dados do arquivo JSON.
    """
    global fingerprint_db
    with open(OUTPUT_JSON, "r") as f:
        fingerprint_db = json.load(f)
    print("Fingerprint carregado com sucesso.")

# --------------------------
# Detecção em tempo real
# --------------------------

live_audio_buffer = np.zeros(WINDOW_SIZE)
detector = FingerprintDetector(fingerprint_db)

def live_matching_callback(indata, frames, time, status):
    global live_audio_buffer

    if status:
        print(status)

    # Buffer circular
    live_audio_buffer = np.roll(live_audio_buffer, -frames)
    live_audio_buffer[-frames:] = indata[:, 0]  # Mono

    features = detector.extract_features(live_audio_buffer)
    best_match = detector.find_best_match(features, window_index=0)

    if best_match[0] is not None:
        print(f"Melhor correspondência: {best_match[0]} | Distância: {best_match[1]:.4f}")

def start_live_detection():
    print("\nIniciando detecção em tempo real... Pressione Ctrl+C para parar.")
    with sd.InputStream(callback=live_matching_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=WINDOW_SIZE):
        while True:
            sd.sleep(100)

# --------------------------
# Execução
# --------------------------

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_JSON):
        generate_fingerprint_database()
    load_fingerprint_database()
    detector.fingerprint_db = fingerprint_db  # recarrega na instância
    start_live_detection()
