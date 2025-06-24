import time
import sounddevice as sd
import soundfile as sf
import numpy as np
from detector import FingerprintDetector

SAMPLE_RATE = 22050
BUFFER_DURATION = 0.1  # 100ms
AUDIO_FILE = "track_22050.wav"

def main():
    """Reproduz o arquivo de áudio sincronizado com a posição detectada."""
    # Carrega o áudio a ser reproduzido
    audio_data, sr = sf.read(AUDIO_FILE)
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Taxa de amostragem do áudio deve ser {SAMPLE_RATE}, mas foi {sr}"
        )

    # Converte para mono se necessário
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    print("Iniciando reprodução sincronizada com vinil...")

    # Inicializa detector
    detector = FingerprintDetector()

    block_size = int(SAMPLE_RATE * BUFFER_DURATION)
    playback_pos = 0  # índice da amostra inicial

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, blocksize=block_size
        ) as input_stream, sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1
        ) as output_stream:
            while True:
                indata, _ = input_stream.read(block_size)
                indata = indata[:, 0]  # mono
                print(
                    f"indata shape: {indata.shape}, min: {indata.min()}, max: {indata.max()}"
                )
                estimated_ts = detector.get_live_position(indata)
                if estimated_ts is not None:
                    print(f"Reproduzindo a partir de {estimated_ts:.2f}s")
                    playback_pos = int(estimated_ts * SAMPLE_RATE)

                end_pos = min(playback_pos + block_size, len(audio_data))
                audio_block = audio_data[playback_pos:end_pos]

                # Garante que o bloco tenha o tamanho esperado
                if len(audio_block) < block_size:
                    audio_block = np.pad(
                        audio_block, (0, block_size - len(audio_block))
                    )

                output_stream.write(audio_block.reshape(-1, 1))
                playback_pos += block_size
                time.sleep(BUFFER_DURATION)

    except KeyboardInterrupt:
        print("\nEncerrado pelo usuário.")


if __name__ == "__main__":
    main()
