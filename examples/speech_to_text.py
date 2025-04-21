from listener import Listener

listener = Listener(
    speech_handler=print,  # replace this with a function of your choice
    on_listening_start=lambda: print("go!"),
    on_speech_start=lambda: print("-- speech detected --"),
)

if __name__ == "__main__":
    try:
        listener.listen(block=True)
        # You'll see transcribed text printed as you speak
    except KeyboardInterrupt:
        listener.close()
