import time
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
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

    # Função da thread de detecção
    def detection_thread():
        nonlocal playback_pos, stop_flag
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
                end_pos = min(playback_pos + block_size, len(audio_data))
                audio_block = audio_data[playback_pos:end_pos]

                # Preenche com zeros se o bloco for pequeno
                if len(audio_block) < block_size:
                    audio_block = np.pad(audio_block, (0, block_size - len(audio_block)))

                output_stream.write(audio_block.reshape(-1, 1))
                playback_pos += block_size
                time.sleep(BUFFER_DURATION)

    except KeyboardInterrupt:
        print("\nEncerrado pelo usuário.")
        stop_flag = True
        thread.join()

if __name__ == "__main__":
    main()
