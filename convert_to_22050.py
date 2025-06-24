import librosa
import soundfile as sf

input_path = "track.wav"  # arquivo original
output_path = "track_22050.wav"

y, sr = librosa.load(input_path, sr=None)
y_resampled = librosa.resample(y, orig_sr=sr, target_sr=22050)

sf.write(output_path, y_resampled, 22050)
print("Arquivo convertido e salvo como track_22050.wav com sr=22050")
