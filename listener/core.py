import math
import multiprocessing as mp
import multiprocessing.shared_memory as shared_mem
import os
import queue
import threading
import time
from multiprocessing.synchronize import Event as EventClass
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pynvml
import sounddevice as sd

from listener.vad import contains_speech

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
ComputeType = Literal[
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "int16",
    "float16",
    "bfloat16",
    "float32",
]


def get_cuda_mem_gbs(device_idx: int):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return mem.free / (1024**3)


def choose_whisper_model(
    device: str, compute: str, en_only: bool
) -> WhisperSize:
    if device.startswith("cuda"):
        if device == "cuda":
            device = "cuda:0"
        avlbl_gbs = get_cuda_mem_gbs(int(device.split(":")[1]))
        if compute != "float32":
            avlbl_gbs = math.ceil(avlbl_gbs)
        if avlbl_gbs >= 10:
            size = "large"
        elif avlbl_gbs >= 6:
            size = "turbo"
        elif avlbl_gbs >= 5:
            size = "medium"
        elif avlbl_gbs >= 2:
            size = "small"
        else:
            size = "base"
    else:
        size = "small" if compute.startswith(("int8", "int16")) else "base"
    if en_only and size not in ("large", "turbo"):
        size += ".en"
    return size


def run_transcriber(
    model_name: str,
    compute_type: Union[ComputeType, str],
    device: str,
    in_que: Union[mp.Queue, queue.Queue],
    out_que: Union[mp.Queue, queue.Queue],
    stop_evt: EventClass,
    show_download: bool,
    whisper_kwargs: Dict[str, Any],
):
    import sys

    import faster_whisper

    kwargs = {"compute_type": compute_type, "device": device}
    kwargs.update(whisper_kwargs)
    if not show_download:
        try:
            devnull = open(os.devnull, "w", encoding="utf-8")
            sys.stdout, sys.stderr = devnull, devnull
            model = faster_whisper.WhisperModel(model_name, **kwargs)
        finally:
            devnull.close()
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    else:
        model = faster_whisper.WhisperModel(model_name, **kwargs)

    while not stop_evt.is_set():
        try:
            arg_tuple = in_que.get(timeout=1)
            if len(arg_tuple) == 1:
                audio = arg_tuple[0]
                segments, _ = model.transcribe(
                    audio=audio, beam_size=5, vad_filter=True
                )
                out_que.put(([seg.text for seg in segments],))
                continue

            shm_name, shape, dtype = arg_tuple
            shm = shared_mem.SharedMemory(name=shm_name)
            audio = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
            segments, _ = model.transcribe(
                audio=audio, beam_size=5, vad_filter=True
            )
            shm.close()
            out_que.put((shm_name, [seg.text for seg in segments]))
        except (KeyboardInterrupt, queue.Empty):
            continue


def transcription_handler(
    out_que: mp.Queue,
    handler: Callable,
    stop_evt: EventClass,
    shared_mem_refs: Optional[Dict[str, shared_mem.SharedMemory]] = None,
    shared_mem_lock: Optional[threading.Lock] = None,
):
    while not stop_evt.is_set():
        try:
            arg_tuple = out_que.get(timeout=1)
            if len(arg_tuple) > 1:
                shm_name, segments = arg_tuple
                with shared_mem_lock:
                    shared_mem_refs[shm_name].close()
                    shared_mem_refs[shm_name].unlink()
                    shared_mem_refs.pop(shm_name)
            else:
                segments = arg_tuple[0]
            handler(segments)
        except (KeyboardInterrupt, queue.Empty):
            continue


class Listener:
    """
    Detects and processes human voice from a stream of audio data.
    """

    def __init__(
        self,
        speech_handler: Optional[Callable[[str], Any]] = None,
        on_listening_start: Optional[Callable] = None,
        sampling_rate: int = 16000,
        time_window: int = 2,
        no_channels: int = 1,
        on_speech_start: Optional[Callable] = None,
        has_voice: Optional[Callable[[np.ndarray], bool]] = None,
        voice_handler: Optional[Callable[[List[np.ndarray]], Any]] = None,
        min_speech_ms: int = 0,
        speech_prob_thresh: float = 0.5,
        whisper_size: Union[WhisperSize, Literal["auto"]] = "auto",
        whisper_kwargs: Optional[Dict[str, Any]] = None,
        compute_type: Optional[Union[ComputeType, str]] = None,
        en_only: bool = False,
        device: Optional[str] = None,
        show_download: bool = True,
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
        speech_handler: Callable[[List[str]], Any], optional
            Function to be called with the text segments extracted from the
            audio with human voice.

        on_listening_start: Callable, optional
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

        min_speech_ms: Minimum number of milliseconds of human voice
            that counts as speech (default: 0).

        speech_prob_thresh: Speech probability less than this will not be
            considered speech (default: 0.5).

        whisper_size: WhisperSize, optional
            Specifies size of the whisper model to be used for converting the
            human voice to text, "pass" auto to let this decision be made
            automatically the best performing model will be chosen based on
            memory availability, note: the model sizes with the '.en' prefix
            are english-only and tend to perform better if the speaker only
            speaks english (default: `"auto"`).

        whisper_kwargs: dict, optional
            Keyword arguments for the underlying `faster_whisper.WhisperModel`
            object.

        compute_type: str, optional
            The data type to use for whisper model's computation
            check https://opennmt.net/CTranslate2/quantization.html for all
            available options.

        en_only: bool, optional
            This flag is used when choosing the optimal whisper model when the
            `whisper_size` argument is not provided, set to `True` if the
            speaker is only going to speak english (default: `False`).

        device: str, optional
            The device to run necessary models on, e.g. cpu, cuda etc
            (default: `cuda` if available, `cpu` otherwise).

        show_download: bool, optional
            Controls whether progress bars are shown while downloading the
            whisper models (default: True).
        """

        assert (voice_handler is None) != (
            speech_handler is None
        ), "pass either 'voice_handler' or 'speech_handler', only one"

        assert (
            0 < speech_prob_thresh <= 1
        ), "speech_prob_thresh must be a value beween '0 < threshold <= 1'"

        if device is None:
            try:
                pynvml.nvmlInit()
                has_cuda = pynvml.nvmlDeviceGetCount() > 0
            except pynvml.NVMLError_LibraryNotFound:
                has_cuda = False
            if has_cuda:
                pynvml.nvmlShutdown()
            device = "cuda" if has_cuda else "cpu"

        if compute_type is None:
            compute_type = "float" + (
                "16" if device.startswith("cuda") else "32"
            )

        whisper_size = (
            whisper_size
            if whisper_size != "auto"
            else choose_whisper_model(device, compute_type, en_only)
        )

        self.speech_handler = speech_handler
        self.on_listening_start = on_listening_start
        self.sampling_rate = sampling_rate
        self.time_window = time_window
        self.no_channels = no_channels
        self.on_speech_start = on_speech_start
        self.has_voice = has_voice
        self.voice_handler = voice_handler
        self.min_speech_ms = min_speech_ms
        self.speech_prob_thresh = speech_prob_thresh
        self.device = device
        self.has_cuda = device.startswith("cuda")
        self.voice_chunks = []
        self.is_paused = False
        self.transcription_mode = speech_handler is not None

        if self.transcription_mode:
            self.que_in_transcriber = (
                queue.Queue() if self.has_cuda else mp.Queue()
            )
            self.que_out_transcriber = (
                queue.Queue() if self.has_cuda else mp.Queue()
            )
            self.evt_stop_transcriber = (
                threading.Event() if self.has_cuda else mp.Event()
            )
            self.evt_stop_transcription_handler = (
                threading.Event() if self.has_cuda else mp.Event()
            )
            self.shared_memory_refs = None if self.has_cuda else {}
            self.shared_memory_lock = (
                None if self.has_cuda else threading.Lock()
            )

            transcriber_args = dict(
                target=run_transcriber,
                args=(
                    whisper_size,
                    compute_type,
                    device,
                    self.que_in_transcriber,
                    self.que_out_transcriber,
                    self.evt_stop_transcriber,
                    show_download,
                    whisper_kwargs if whisper_kwargs is not None else {},
                ),
            )
            if self.has_cuda:
                self.transcriber = threading.Thread(**transcriber_args)
            else:
                self.transcriber = mp.Process(**transcriber_args)
            self.transcription_handler = threading.Thread(
                target=transcription_handler,
                args=(
                    self.que_out_transcriber,
                    self.speech_handler,
                    self.evt_stop_transcription_handler,
                    self.shared_memory_refs,
                    self.shared_memory_lock,
                ),
            )

        if self.has_voice is None:

            def has_voice(audio: np.ndarray):
                return contains_speech(
                    audio,
                    sampling_rate=self.sampling_rate,
                    min_speech_ms=self.min_speech_ms,
                    speech_prob_thresh=self.speech_prob_thresh,
                )

            self.has_voice = has_voice

    def _audio_cb(self, in_data: np.ndarray, *args):
        if self.is_paused:
            return
        if (
            self.transcription_mode
            and self.evt_stop_transcription_handler.is_set()
        ):
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
            if self.transcription_mode:
                frames = np.concatenate(self.voice_chunks)
                if self.has_cuda or os.name != "nt":
                    self.que_in_transcriber.put((frames.ravel(),))
                else:
                    shm = shared_mem.SharedMemory(
                        create=True, size=frames.size * frames.itemsize
                    )
                    shared_arr = np.ndarray(
                        shape=(frames.size,),
                        dtype=frames.dtype,
                        buffer=shm.buf,
                    )
                    shared_arr[:] = frames.ravel()
                    self.que_in_transcriber.put(
                        (shm.name, frames.shape, frames.dtype)
                    )
                    # this reference keeps the shared memory from
                    # being garbage collected
                    with self.shared_memory_lock:
                        self.shared_memory_refs[shm.name] = shm
            else:
                self.voice_handler(self.voice_chunks)
            self.voice_chunks.clear()

    def reset(self):
        self.voice_chunks = []
        if not self.transcription_mode:
            return
        self.evt_stop_transcriber.clear()
        self.evt_stop_transcription_handler.clear()

    def listen(self, block: bool = False):
        """
        Starts listening from a separate thread.
        """
        self.stream = sd.InputStream(
            samplerate=self.sampling_rate,
            blocksize=self.sampling_rate * self.time_window,
            channels=self.no_channels,
            callback=self._audio_cb,
        )

        self.reset()
        if self.transcription_mode:
            self.transcriber.start()
            self.transcription_handler.start()
        self.stream.start()
        if self.on_listening_start is not None:
            self.on_listening_start()
        if not block:
            return
        while True:
            time.sleep(1)

    def pause(self):
        """
        Pauses listening until `resume` is called.
        """
        self.is_paused = True

    def resume(self):
        """
        Resumes listening after `pause` is called.
        """
        self.is_paused = False

    def close(self):
        """
        Stops listening and frees held resources.
        """
        if self.transcription_mode:
            self.evt_stop_transcriber.set()
            self.evt_stop_transcription_handler.set()
            self.transcriber.join()
            self.transcription_handler.join()
            if self.shared_memory_refs is not None:
                with self.shared_memory_lock:
                    for name in self.shared_memory_refs.keys():
                        self.shared_memory_refs[name].close()
                        self.shared_memory_refs[name].unlink()
                    self.shared_memory_refs.clear()
        self.voice_chunks.clear()
