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
listener = Listener(speech_handler=print)

# start listening
listener.listen()

# NOTE: listening is done from a separate thread, so you must
# have other operations to keep the interpreter going, or it will
# quit. if your code has no other operations, just run a loop like
# below.

# --------------------
# import time

# while True:
#     time.sleep(1)
# -----------------------

# stops listening
listener.stop()

# starts listening again
# listener.listen()
```

## Documentation
There is only one class in the package, the `Listener`.

It starts collecting audio data after instantation into `n` second chunks, `n` is a number passed as an argument, it checks if the audio chunk contains any human voice in it and if there is human voice, it collects that chunk for later processing (conversion to text or any other user-defined processing) and discards the chunk otherwise.

#### Constructor parameters
- `speech_handler`: a function that is called with the text for the human voice in the recorded audio as the only argument, `speech_handler(string speech)`.

- `on_listening_start`: a parameterless function that is called right after the Listener object starts collecting audio.

- `time_window`: an integer that specifies the chunk size of the collected audio in seconds, `2` is the default.

- `no_channels`: the number of audio channels to be used for recording, `1` is the default.

- `has_voice`: a function that is called on the recorded audio chunks to determine if they have human voice in them, it gets the audio chunk in a `numpy.ndarray` object as the only argument, [Silero](https://github.com/snakers4/silero-vad) is used by default to do this, `has_voice(numpy.ndarrray chunk)`.

- `voice_handler`: a function that is used to process [an utterance](https://en.wikipedia.org/wiki/Utterance), a continuous segment of speech, it gets a list of audio chunks as the only argument, `voice_handler(list<numpy.ndarray>)`.

- `voice_to_speech`: a function used to convert human voice to text, [whisper](https://github.com/openai/whisper) is used by default to do this, `voice_to_speech(list<numpy.ndarray>)`.

- `use_fp16`: a boolean flag indicating if the the voice detection and speech-to-text models should use [half precision](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) arithmetic to save memory and reduce latency, the default is `True` if CUDA is available, it has no effect on CPUs at the time of this writing so it's set to `False` by default on CPU environments.

- `en_only`: a flag indicating only english language is going to be used in the collected audio, this is used to determine the best whisper model to use to convert speech to text.

- `device`: this the device where the speech detection and speech to text conversion models run, the default is `cuda if available, else cpu`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
