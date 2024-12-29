# Listener

Listener is a Python library for real-time speech to text conversion.

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

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
