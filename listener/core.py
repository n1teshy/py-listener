import hashlib
import math
import os
import threading
import urllib
from typing import Any, Callable, List, Literal, Optional, Union

import numpy as np
import sounddevice as sd
import torch

from listener.utils import whisper_source

WhisperSize = Literal[
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
    "turbo",
]


def choose_whisper_model(
    device: torch.device, use_fp16: bool, en_only: bool
) -> WhisperSize:
    if device.type.lower() in ("cpu", "mps"):
        return "small.en" if en_only else "small"
    total = torch.cuda.get_device_properties(device).total_memory
    avlbl_gb = (total - torch.cuda.memory_allocated(device)) / (1024**3)
    if use_fp16:
        # assuming fp16 will reduce model's memory consumption
        avlbl_gb = math.ceil(avlbl_gb)
    if avlbl_gb >= 10:
        size = "large"
    elif avlbl_gb >= 6:
        size = "turbo"
    elif avlbl_gb >= 5:
        size = "medium"
    elif avlbl_gb >= 2:
        size = "small"
    else:
        size = "base"
    if en_only and size not in ("large", "turbo"):
        size += ".en"
    return size


def download_whisper_model(size: str):
    url = whisper_source[size]
    default = os.path.join(os.path.expanduser("~"), ".cache")
    root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")

    expected_sha256 = url.split("/")[-2]
    model_file = os.path.join(root, os.path.basename(url))

    if os.path.exists(model_file) and not os.path.isfile(model_file):
        raise RuntimeError(f"{model_file} exists and is not a regular file")

    if os.path.isfile(model_file):
        with open(model_file, "rb") as f:
            model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return

    with urllib.request.urlopen(url) as source, open(
        model_file, "wb"
    ) as output:
        while True:
            buffer = source.read(8192)
            if not buffer:
                break
            output.write(buffer)
    model_bytes = open(model_file, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum "
            "does not not match. Please retry loading the model."
        )


class Listener:
    """
    Detects and processes human voice from a stream of audio data.
    """

    def __init__(
        self,
        speech_handler: Optional[Callable[[str], Any]] = None,
        on_istenining_start: Optional[Callable] = None,
        sampling_rate: int = 16000,
        time_window: int = 2,
        no_channels: int = 1,
        on_speech_start: Optional[Callable] = None,
        has_voice: Optional[Callable[[np.ndarray], bool]] = None,
        voice_handler: Optional[Callable[[List[np.ndarray]], Any]] = None,
        voice_to_speech: Optional[Callable[[List[np.ndarray]], Any]] = None,
        whisper_size: Union[WhisperSize, Literal["auto"]] = "auto",
        use_fp16: Optional[bool] = None,
        en_only: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Collects audio data in `time_window` second chunks and when human
        voice is detected keeps collecting audio chunks until a chunk without
        human voice is found, at that point, it is assumed that the speaker
        is done speaking and one of the following things happen.

        - if `voice_handler` is passed, `voice_handler` is called with the
          audio containing human voice.
        - if `speech_handler` is passed, the collected audio is converted to
          text using `openai-whisper` and `speech_handler` is called with this
          text.

        Parameters
        ----------
        speech_handler: Callable[[str], Any], optional
            Function to be called with the text extracted from the audio with
            human voice.

        on_istenining_start: Callable, optional
            Function to call when `Listener` starts listening.

        sampling_rate : int, optional
            The sampling rate (hertz) to be used for capturing sound
            (default: `44100` or 44.1 KHz).

        time_window: int, optional
            The number of seconds of audio to collect before checking if it
            contains human voice (default: `2`).
            - - larger values mean: the speaker will not have to rush through
                whatever they have to say to fit it into the narrow window of
                `time_window` seconds and it may also be easier on the CPU
                since there are less chunks to analyze.
            - - smaller values maean: better responsiveness e.g. if you say
                "hello" and the time_window is 5 seconds, you would have to
                wait extra ~4 seconds before your voice is even detected, a
                smaller `time_window` is preferrble in this case, but it may
                also take more CPU time since there are more chunks to analyze.

        no_channels : int, optional
            Number of audio channels to be used for recording (default: `1`).

        on_speech_start: Callable, optional
            Function to be called when the speaker starts speaking.

        has_voice: Callable[[np.ndarray], bool], optional
            User defined function to determine if a chunk if audio contains
            human voice.

        voice_handler: Callable[[List[np.ndarray]], Any], optional
            Function to be called with the collected audio when speaker is
            done speaking.

        voice_to_speech: Callable[[List[np.ndarray]], Any], optional
            User defined function to convert collected audio with human voice
            to text, a speech-to-text function.

        whisper_size: WhisperSize, optional
            Specifies size of the whisper model to be used for converting the
            human voice to text, "pass" auto to let this decision be made
            automatically the best performing model will be chosen based on
            memory availability, note: the model sizes with the '.en' prefix
            are english-only and tend to perform better if the speaker only
            speaks english (default: `"auto"`).

        use_fp16: bool, optional
            Specifies if the whisper model should use half-precision (16 bit
            floats) arithmetic instead of the typical single-precision (32 bit
            floats), this reduces model's memory footprint and lowers latency
            (default: False for CPU and True for GPU).

        en_only: bool, optional
            This flag is used when choosing the optimal whisper model when the
            `whisper_size` argument is not provided, set to `True` if the
            speaker is only going to speak english (default: `False`).

        device: Union[str, torch.device], optional
            The device to run necessary models on, e.g. cpu, cuda etc
            (default: `"cuda"` if available, `"cpu"` otherwise).
        """

        assert (voice_handler is None) != (
            speech_handler is None
        ), "pass either 'voice_handler' or 'speech_handler', only one"
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            device = torch.device(device) if type(device) is str else device
        if use_fp16 is None:
            use_fp16 = device.type.lower() not in ("cpu", "mps")
        self.speech_handler = speech_handler
        self.on_istenining_start = on_istenining_start
        self.sampling_rate = sampling_rate
        self.time_window = time_window
        self.no_channels = no_channels
        self.on_speech_start = on_speech_start
        self.has_voice = has_voice
        self.voice_handler = voice_handler
        self.voice_to_speech = voice_to_speech
        self.whisper_size = (
            whisper_size
            if whisper_size != "auto"
            else choose_whisper_model(device, use_fp16, en_only)
        )
        self.use_fp16 = use_fp16
        self.device = device
        self.voice_chunks = []
        self._stop = threading.Event()
        self._free_cuda = False
        # openai-whisper shows a progress bar when it downloads a model
        # I don't like the bar messing up my terminal, this is a temporary fix
        if self.speech_handler is not None and self.voice_to_speech is None:
            download_whisper_model(self.whisper_size)

    def listen(self):
        self.stream = sd.InputStream(
            samplerate=self.sampling_rate,
            blocksize=self.sampling_rate * self.time_window,
            channels=self.no_channels,
            callback=self._audio_cb,
        )
        if self.has_voice is None:
            vad_model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
            )
            vad_model = vad_model.to(self.device)
            get_speech_timestamps = utils[0]

            def is_voice(frames: np.ndarray):
                frames = torch.tensor(
                    frames, dtype=torch.float32, device=self.device
                )
                timestamps = get_speech_timestamps(
                    frames, vad_model, sampling_rate=self.sampling_rate
                )
                return len(timestamps) > 0

            self.has_voice = is_voice

        if self.speech_handler is not None and self.voice_to_speech is None:
            import whisper

            model = whisper.load_model(self.whisper_size, self.device)

            def voice_to_speech(frames: List[np.ndarray]) -> str:
                audio = torch.from_numpy(np.concatenate(frames)).to(
                    model.device
                )
                return model.transcribe(audio, fp16=self.use_fp16)["text"]

            self.voice_to_speech = voice_to_speech
            self._free_cuda = True

        self._stop.clear()
        self.stream.start()
        if self.on_istenining_start is not None:
            self.on_istenining_start()

    def _audio_cb(self, in_data: np.ndarray, *args):
        if self._stop.is_set():
            self.stream.stop()
            return
        frames = in_data.flatten()
        if self.has_voice(frames):
            if (
                len(self.voice_chunks) == 0
                and self.on_speech_start is not None
            ):
                self.on_speech_start()
            self.voice_chunks.append(frames)
        elif len(self.voice_chunks) > 0:
            if self.voice_handler is not None:
                self.voice_handler(self.voice_chunks)
            else:
                self.speech_handler(self.voice_to_speech(self.voice_chunks))
            self.voice_chunks = []

    def stop(self):
        self._stop.set()
        self.voice_chunks = []
        # free the whisper model from the reference held by
        # self.voice_to_speech() so it can be grabage collected
        if self._free_cuda:
            del self.voice_to_speech
            self.voice_to_speech = None
            torch.cuda.empty_cache()
