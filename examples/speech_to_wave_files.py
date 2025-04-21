import wave

import numpy as np

from listener import Listener

files_written = 0


def to_file(chunks: list[np.ndarray]):
    global files_written

    audio_data = np.concatenate(chunks)
    # convert float32 audio samples to PCM(int16) format
    audio_data = (audio_data * 32768).astype(np.int16)

    with wave.open(f"{files_written}.wav", "wb") as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(16000)
        wave_file.writeframes(audio_data.tobytes())

    files_written += 1


if __name__ == "__main__":
    try:
        listener = Listener(
            voice_handler=to_file,
            on_speech_start=lambda: print("--- speech detected ---"),
            on_listening_start=lambda: print("listening"),
        )
        listener.listen(block=True)
    except KeyboardInterrupt:
        listener.close()
