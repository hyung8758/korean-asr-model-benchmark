import tempfile
import wave
from pathlib import Path


TARGET_SAMPLE_RATE = 16000


def save_upload(data: bytes, suffix: str = ".wav") -> Path:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        handle.write(data)
        return Path(handle.name)
    finally:
        handle.close()


def wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            if sample_rate <= 0:
                return None
            return frames / float(sample_rate)
    except wave.Error:
        return None


def read_wav_chunk(data: bytes) -> tuple[bytes, int, int, int]:
    path = save_upload(data, ".wav")
    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
            return frames, channels, sample_width, sample_rate
    finally:
        path.unlink(missing_ok=True)


def write_wav(path: Path, pcm: bytes, channels: int, sample_width: int, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)

