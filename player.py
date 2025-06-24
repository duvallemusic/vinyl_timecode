
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading

# Caminho do arquivo de áudio a ser tocado
AUDIO_FILE = "track.wav"
BUFFER_SIZE = 1024

class AudioPlayer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data, self.samplerate = sf.read(self.file_path, dtype='float32')
        self.position = 0
        self.playing = True
        self.lock = threading.Lock()

    def audio_callback(self, outdata, frames, time, status):
        with self.lock:
            if not self.playing:
                outdata[:] = np.zeros((frames, self.data.shape[1]))
                return
            end = self.position + frames
            if end > len(self.data):
                end = len(self.data)
                self.playing = False
            out_chunk = self.data[self.position:end]
            if len(out_chunk) < frames:
                out_chunk = np.pad(out_chunk, ((0, frames - len(out_chunk)), (0, 0)), mode='constant')
            outdata[:] = out_chunk
            self.position += frames

    def play(self):
        with sd.OutputStream(
            samplerate=self.samplerate,
            channels=self.data.shape[1],
            callback=self.audio_callback,
            blocksize=BUFFER_SIZE
        ):
            print("Reproduzindo... Comandos: [p] play/pause, [r] reiniciar, [s] sair")
            while True:
                cmd = input(">> ").strip().lower()
                with self.lock:
                    if cmd == "p":
                        self.playing = not self.playing
                        print("Play" if self.playing else "Pause")
                    elif cmd == "r":
                        self.position = 0
                        print("Reiniciado")
                    elif cmd == "s":
                        break

if __name__ == "__main__":
    player = AudioPlayer(AUDIO_FILE)
    player.play()
