# The code below is adapted from https://github.com/SYSTRAN/faster-whisper.

import functools
import os

import numpy as np

from listener.utils import get_assets_path


@functools.lru_cache
def get_vad_model():
    """Returns the VAD model instance."""
    encoder_path = os.path.join(get_assets_path(), "silero_encoder_v5.onnx")
    decoder_path = os.path.join(get_assets_path(), "silero_decoder_v5.onnx")
    return SileroVADModel(encoder_path, decoder_path)


def contains_speech(
    audio: np.ndarray,
    sampling_rate: int,
    min_speech_ms: int,
    speech_prob_thresh: float,
) -> bool:
    """This method is used for splitting long audios into speech chunks using
    silero VAD.

    Parameters
    ----------
    audio: One dimensional float array represending audio.
    sampling_rate: Sampling rate of the audio being processed.
    min_speech_ms: Minimum number of milliseconds of human voice
        that counts as speech.
    speech_prob_thresh: Speech probability less than this will not be
        considered speech.
    """
    window_size_samples = 512
    model = get_vad_model()
    padded_audio = np.pad(
        audio, (0, window_size_samples - audio.shape[0] % window_size_samples)
    )
    speech_probs = model(padded_audio.reshape(1, -1)).reshape(-1)
    last_speech_idx, max_segment = None, 0
    for idx, prob in enumerate(speech_probs):
        if prob > speech_prob_thresh:
            if last_speech_idx is None:
                last_speech_idx = idx
        elif last_speech_idx is not None:
            max_segment = max(idx - last_speech_idx, max_segment)
            last_speech_idx = None
    max_speech_samples = max_segment * (audio.size / speech_probs.size)
    return int(max_speech_samples / sampling_rate * 1000) > min_speech_ms


class SileroVADModel:
    def __init__(self, encoder_path, decoder_path):
        try:
            import onnxruntime
        except ImportError as e:
            raise RuntimeError(
                "Applying the VAD filter requires the onnxruntime package"
            ) from e

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 4

        self.encoder_session = onnxruntime.InferenceSession(
            encoder_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self.decoder_session = onnxruntime.InferenceSession(
            decoder_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )

    def __call__(
        self,
        audio: np.ndarray,
        num_samples: int = 512,
        context_size_samples: int = 64,
    ):
        assert (
            audio.ndim == 2
        ), "Input should be a 2D array with size (batch_size, num_samples)"
        assert (
            audio.shape[1] % num_samples == 0
        ), "Input size should be a multiple of num_samples"
        batch_size = audio.shape[0]

        state = np.zeros((2, batch_size, 128), dtype="float32")
        context = np.zeros(
            (batch_size, context_size_samples),
            dtype="float32",
        )

        batched_audio = audio.reshape(batch_size, -1, num_samples)
        context = batched_audio[..., -context_size_samples:]
        context[:, -1] = 0
        context = np.roll(context, 1, 1)
        batched_audio = np.concatenate([context, batched_audio], 2)

        batched_audio = batched_audio.reshape(
            -1, num_samples + context_size_samples
        )

        encoder_batch_size = 10000
        num_segments = batched_audio.shape[0]
        encoder_outputs = []
        for i in range(0, num_segments, encoder_batch_size):
            encoder_output = self.encoder_session.run(
                None, {"input": batched_audio[i: i + encoder_batch_size]}
            )[0]
            encoder_outputs.append(encoder_output)

        encoder_output = np.concatenate(encoder_outputs, axis=0)
        encoder_output = encoder_output.reshape(batch_size, -1, 128)
        decoder_outputs = []
        for window in np.split(
            encoder_output, encoder_output.shape[1], axis=1
        ):
            out, state = self.decoder_session.run(
                None, {"input": window.squeeze(1), "state": state}
            )
            decoder_outputs.append(out)
        out = np.stack(decoder_outputs, axis=1).squeeze(-1)
        return out
