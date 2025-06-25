import time
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
from detector import FingerprintDetector

SAMPLE_RATE = 22050
BUFFER_DURATION = 0.1  # 100ms
AUDIO_FILE = "track_22050.wav"

def main():
    print("Iniciando reprodução sincronizada com vinil...")

    # Carrega o áudio a ser reproduzido
    audio_data, sr = sf.read(AUDIO_FILE)
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Taxa de amostragem do áudio deve ser {SAMPLE_RATE}, mas foi {sr}"
        )

    # Converte para mono se necessário
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Inicializa detector e variáveis
    detector = FingerprintDetector()
    block_size = int(SAMPLE_RATE * BUFFER_DURATION)
    playback_pos = 0
    stop_flag = False
    playback_speed = 1.0
    direction = 1

    # Função da thread de detecção
    def detection_thread():
        nonlocal playback_pos, stop_flag, playback_speed, direction
        previous_ts = None
        previous_time = None

        while not stop_flag:
            try:
                block, _ = input_stream.read(block_size)
                mono_data = block[:, 0]

                # Ignora buffers silenciosos
                if len(mono_data) == 0 or np.max(np.abs(mono_data)) < 1e-5:
                    continue

                result = detector.get_live_position(mono_data)
                if result is not None:
                    estimated_ts, _ = result
                    now = time.time()

                    if previous_ts is not None and previous_time is not None:
                        dt_pos = estimated_ts - previous_ts
                        dt_time = now - previous_time
                        if dt_time > 0:
                            new_speed = dt_pos / dt_time
                            # suaviza a estimativa de velocidade
                            playback_speed = 0.7 * playback_speed + 0.3 * abs(new_speed)
                            direction = -1 if new_speed < 0 else 1

                    previous_ts = estimated_ts
                    previous_time = now

                    print(f"Reproduzindo a partir de {estimated_ts:.2f}s")
                    playback_pos = int(estimated_ts * SAMPLE_RATE)
            except Exception as e:
                print("Erro na thread de detecção:", e)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, blocksize=block_size
        ) as input_stream, sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1
        ) as output_stream:

            print("Reproduzindo... Pressione Ctrl+C para parar.")

            # Inicia thread paralela
            thread = threading.Thread(target=detection_thread)
            thread.start()

            # Loop principal de reprodução
            while True:
                current_size = int(block_size * abs(playback_speed)) or block_size

                if direction == 1:
                    end_pos = min(playback_pos + current_size, len(audio_data))
                    audio_block = audio_data[playback_pos:end_pos]
                    playback_pos = end_pos
                else:
                    start_pos = max(playback_pos - current_size, 0)
                    audio_block = audio_data[start_pos:playback_pos][::-1]
                    playback_pos = start_pos

                if len(audio_block) == 0:
                    audio_block = np.zeros(block_size)
                else:
                    try:
                        audio_block = librosa.effects.time_stretch(audio_block, abs(playback_speed))
                    except Exception:
                        pass

                if len(audio_block) < block_size:
                    audio_block = np.pad(audio_block, (0, block_size - len(audio_block)))
                elif len(audio_block) > block_size:
                    audio_block = audio_block[:block_size]

                output_stream.write(audio_block.reshape(-1, 1))
                time.sleep(BUFFER_DURATION)

    except KeyboardInterrupt:
        print("\nEncerrado pelo usuário.")
        stop_flag = True
        thread.join()

if __name__ == "__main__":
    main()
