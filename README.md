
- [hybrid-ssvep-p300-speller](#hybrid-ssvep-p300-speller)
- [Set up the project.](#set-up-the-project)
- [About BrainFlow](#about-brainflow)
- [Getting Start](#getting-start)

# hybrid-ssvep-p300-speller

# Set up the project.

To set this project up, this is what you need.

1. Python 3.8.10 [link](https://www.python.org/downloads/release/python-3810/)
2. Windows PC/Laptop (We tested this on Windows 10/11)
3. pipenv

What you have to do is

1. Download and install python 3.8.10.
2. Open this project (we recommend VScode)
3. Install `pipenv` using `pip install pipenv`
4. At the root of the project, run `pipenv install`
5. Wait...
6. Done.


Happy coding

# About BrainFlow

Our project relies on `BrainFlow` library to get data stream from the EEG hat/board.
We recommend you to read a bit on their document.
For starter, we used `G.TEC UNICORN` to record EEG signals.
This is how you initialize the hat [link](https://brainflow.readthedocs.io/en/stable/SupportedBoards.html#unicorn).

`BrainFlow` also provided you a dummy board.
There are three kinds of them but we only use two.
And during development, you may use this `Synthetic Board` [link](https://brainflow.readthedocs.io/en/stable/SupportedBoards.html#synthetic-board).
To rerun the experiment with existing record, you can use `Playback File Board` [link](https://brainflow.readthedocs.io/en/stable/SupportedBoards.html#playback-file-board).

# Getting Start

You can start by reading the `_get_start` folder.