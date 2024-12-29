from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent.resolve()
meta = {}
exec((ROOT / "listener/meta.py").read_text(), meta)
NAME, VERSION, GITHUB = meta["name"], meta["version"], meta["github"]
long_description = (ROOT / "README.md").read_text()


setup(
    name=NAME,
    version=VERSION,
    author="Nitesh Yadav",
    author_email="nitesh.txt@gmail.com",
    description="Real time speech to text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=GITHUB,
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchaudio",
        "numpy",
        "sounddevice",
        "openai-whisper",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    platforms="Posix; Windows",
    keywords=[
        "speech-to-text",
        "offline speech to text",
        "stt",
    ],
    python_requires=">=3.8",
    license="MIT",
)
