# Listener

Listener is a Python library for real-time speech to text conversion.

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

# stops listening
listener.stop()

# start listening again
# listener.listen()
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
