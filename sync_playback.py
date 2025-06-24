import time
import sounddevice as sd
import soundfile as sf
import numpy as np
from detector import FingerprintDetector

SAMPLE_RATE = 22050
BUFFER_DURATION = 0.1  # 100ms
AUDIO_FILE = "track_22050.wav"

# Carrega o áudio a ser reproduzido
audio_data, sr = sf.read(AUDIO_FILE)
if sr != SAMPLE_RATE:
    raise ValueError(f"Taxa de amostragem do áudio deve ser {SAMPLE_RATE}, mas foi {sr}")

# Converte para mono se necessário
if audio_data.ndim > 1:
    audio_data = np.mean(audio_data, axis=1)

print("Iniciando reprodução sincronizada com vinil...")

# Inicializa detector
detector = FingerprintDetector()

block_size = int(SAMPLE_RATE * BUFFER_DURATION)
playback_pos = 0  # índice da amostra inicial

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=block_size) as input_stream, \
         sd.OutputStream(samplerate=SAMPLE_RATE, channels=1) as output_stream:
        while True:
            indata, _ = input_stream.read(block_size)
            indata = indata[:, 0]  # mono
            print(f"indata shape: {indata.shape}, min: {indata.min()}, max: {indata.max()}")
            estimated_ts = detector.get_live_position(indata)
            if estimated_ts is not None:
                print(f"Reproduzindo a partir de {estimated_ts:.2f}s")
                playback_pos = int(estimated_ts * SAMPLE_RATE)

            end_pos = min(playback_pos + block_size, len(audio_data))
            audio_block = audio_data[playback_pos:end_pos]

            # Garante que o bloco tenha o tamanho esperado
            if len(audio_block) < block_size:
                audio_block = np.pad(audio_block, (0, block_size - len(audio_block)))

            output_stream.write(audio_block.reshape(-1, 1))
            playback_pos += block_size
            time.sleep(BUFFER_DURATION)

except KeyboardInterrupt:
    print("\nEncerrado pelo usuário.")

# No extract_features
assert mfcc.size > 0, "MFCC vazio!"

from scipy.spatial.distance import cosine

def compare_features(self, live, feature_list):
    # Verifica se MFCC está vazio
    if len(live["mfcc"]) == 0 or len(feature_list[0]["mfcc"]) == 0:
        return float("inf")
    for f in feature_list:
        # espera-se que f seja um dict, mas pode ser uma string
        if isinstance(f, dict) and "mfcc" in f:
            distance = cosine(live["mfcc"], f["mfcc"])
            if distance < min_distance:
                min_distance = distance
                best_match = f
    return best_match

# Defina o valor de 'track' antes de usá-lo, por exemplo:
track = "track_22050.wav"  # Substitua pelo nome correto do track
variation = "nome_da_variacao"  # Substitua pelo nome correto da variação
# Substitua 'obj' pelo nome do objeto que possui o atributo 'fingerprint_db'
# Por exemplo, se você tem detector = FingerprintDetector(), use detector.fingerprint_db
feature_list = list(detector.fingerprint_db[track][variation].values())
