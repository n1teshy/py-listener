# PyListener

PyListener is tool for near real-time voice processing and speech to text conversion, it can be pretty
fast to slightly sluggish depending on the compute and memory availability of the environment, I suggest
using it in situations where a delay of ~1 second is reasonable, e.g. AI assistants, voice command
processing etc.

[![Watch a demo](https://img.youtube.com/vi/SEFm8rJRg_A/0.jpg)](https://www.youtube.com/watch?v=SEFm8rJRg_A)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install py-listener
```

## Basic Usage

```python
from listener import Listener

# prints what the speaker is saying, look at all
# parameters of the constructor to find out more features
try:
    listener = Listener(speech_handler=print)
except:
    listener.close()
    raise

# start listening
listener.listen(block=True)

# stops listening
# listener.close()

# starts listening again
# listener.listen()
```

## Documentation
There is only one class in the package, the `Listener`.

It starts collecting audio data after instantation into `n` second chunks, `n` is a number passed as an argument, it checks if the audio chunk contains any human voice in it and if there is human voice, it collects that chunk for later processing (conversion to text or any other user-defined processing) and discards the chunk otherwise.

#### Constructor parameters
- `speech_handler`: a function that is called with the text segments extracted from the human voice in the recorded audio as the only argument, `speech_handler(speech: List[str])`.

- `on_listening_start`: a parameterless function that is called right after the Listener object starts collecting audio.

- `time_window`: an integer that specifies the chunk size of the collected audio in seconds, `2` is the default.

- `no_channels`: the number of audio channels to be used for recording, `1` is the default.

- `has_voice`: a function that is called on the recorded audio chunks to determine if they have human voice in them, it gets the audio chunk in a `numpy.ndarray` object as the only argument, [Silero](https://github.com/snakers4/silero-vad) is used by default to do this, `has_voice(chunk: numpy.ndarrray)`.

- `voice_handler`: a function that is used to process [an utterance](https://en.wikipedia.org/wiki/Utterance), a continuous segment of speech, it gets a list of audio chunks as the only argument, `voice_handler(List[numpy.ndarray])`.

- `min_speech_ms`: the minimum number of miliseconds of voice activity that counts as speech, default is `0`, that means all human voice activity is considered valid speech, no matter how brief.

- `speech_prob_thresh`: the minimum probability of voice activity that is actually considered human voice, must be `0-1`, default is `0.5`.

- `whisper_size`: a size for the underlying whisper model, options: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`,
`medium`, `medium.en`, `large`, `turbo`.
f
- `whisper_kwargs`: keyword arguments for the underlying `faster_whisper.WhisperModel` instance see [faster_whisper](https://github.com/SYSTRAN/faster-whisper) for all available arguments.

- `compute_type`: compute type for the whisper model, options: `int8`, `int8_float32`, `int8_float16s`, `int8_bfloat16`, `int16`, `float16`, `bfloat16`, `float32`, check [OpenNMT docs](https://opennmt.net/CTranslate2/quantization.html) for an updated list of options.

- `en_only`: a boolean flag indicating if the the voice detection and speech-to-text models should use [half precision](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) arithmetic to save memory and reduce latency, the default is `True` if CUDA is available, it has no effect on CPUs at the time of this writing so it's set to `False` by default on CPU environments.

- `device`: this the device where the speech detection and speech to text conversion models run, the default is `cuda if available, else cpu`.

- `show_download`: controls whether progress bars are shown while downloading the whisper models, default is `True`.

### Methods
- **`listen`**: starts recording audio and
- - if a `voice_handler` function is passed, calls the function with the accumulated human voice, `voice_handler(list[numpy.ndarray])`.
- - else, it keeps recording human voice in `time_window` second chunks until a `time_window` second long silence is detected, at which point it converts the accumulated voice to text, and calls the given `speech_handler` function and passes this transcription to it as the only argument, the transcription is a list of text segments, `speech_handler(List[str])`.

- **`pause`**: pauses listening without clearing up held resources.

- **`resume`**: resumes listening if it was paused.

- **`close`**: stops recording audio and frees the resource held by the listener.

- - **NOTE: it is imperative to call the `close` method when the listener is no longer required, because on CPU-only systems, the audio is sent to the model running in the child process in shared memory objects and if the `close` method is not called, thos
objects may not get cleared and result in memory leaks, it also terminates the child process.**

### Tasks

- [ ] use shared memory to pass audio to child process on POSIX
- - issue: resource tracker keeps tracking shared memory object even if instructed not to do so since shared memory is cleared by library's code
- [ ] in `core.py -> choose_whisper_model()` method, allow larger models as `compute_type` gets smaller, the logic is too rigid at the moment.


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
