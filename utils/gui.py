"""
Functions for the experimental tasks

Author: Ludovic Darmet; Juan Jesus Torre Tresols
Mail: ludovic.darmet@isae-supaero.fr; Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import platform
import scipy.signal
import sys
import warnings
import os
import sys
path = os.path.dirname(os.path.dirname(__file__)) 
sys.path.append(path)
from utils.speller_config import *


import numpy as np

from psychopy import visual

if platform.system() == "Linux":
    import re
    import subprocess
elif platform.system() == "Windows":
    import win32api
else:
    print("It looks like you are running this on a system that is not Windows or Linux. \
           Your system is not supported for running the experiment, sorry!")
    sys.exit()

class CheckerBoard:
    """
    Square visual checherboard stimuli for Psychopy.

    Parameters
    ----------

    window: psychopy.visual.Window instance
            Window where to allocate the checkerboard

    size: int
          Size of each square's sides, in psychopy units

    rows: int
          Number of rows
    
    columns: int
             Number of columns

    frequency: int
               Frequency of flicker, in Hz

    phase: float
           Phase of flicker

    amplitude: float
               Amplitude of the sinewave

    wave_type: str, ['sin', 'square'] default 'sin'
               Parameter controlling the type of wave that will set the opacity of the stimuli

    duration: int, default 3
              Total duration of one flicker presentation in seconds

    fps: int, default 60
         Frames per second or refresh rate of the monitor, in Hz/frames

    base_pos: tuple of int, default (0, 0)
              Base position for the stimuli to draw. This represents an absolute offset that
              will be applied to the whole array of squares. Defaults to (0, 0), which corresponds
              to a position on the center of the screen

    Attributes
    ----------

    window: psychopy.visual.Window instance
            Window where to allocate the checkerboard

    size: int
          Size of each square's sides, in psychopy units


    rows: int
          Number of rows
    
    columns: int
             Number of columns

    total_size: tuple of int
                Total size of the checkerboard, taking into account rows and columns

    freq: int
          Frequency of flicker, in Hz

    phase: float
           Phase of flicker

    amp: float
         Amplitude of the flicker sinewave

    wave_type: numpy.sin or scipy.signal.square function
               Function to create the signal for the stimuli

    func_values: np.array
                 Wave corresponding to the values of the flicker over time. Only used in
                 the draw2 method

    bound_pos: dict of int
               Dictionary containing the boundary positions within the screen where the stimuli
               can be displayed without being cut

    base_x: int
            Absolute offset on the X axis (rows) for the whole array of stim
    
    base_y: int
            Absolute offset on the Y axis (columns) for the whole array of stim

    board: np.array of psychopy.Rect objects
           Array of rows x columns shape, with each
           element being the rect instance that corresponds
           to that position in the checkerboard
    
    """

    def __init__(self, window, size=1000, rows=1, columns=1, frequency=10, phase=0, amplitude=1., wave_type='sin', duration=2.2, fps=60, base_pos=(0, 0), height=100, width=100):
        self.window = window
        self.size = size
        self.rows = rows
        self.cols = columns
        self.total_size = self.rows * self.size, self.cols * self.size
        self.freq = frequency
        self.phase = phase
        self.amp = amplitude
        self.wave_type = wave_type
        self.height = height
        self.width = width

        self.wave_func = self._get_wave_func(duration, fps)
        self.bound_pos = self._get_boundaries()
        self.base_x, self.base_y = self._check_base_pos(base_pos)
        self.board = self._make_board()

    @staticmethod
    def _alternate_color(current_color):
        """Alternate between black and white strings"""

        if current_color == 'black': 
            color = 'white'
        elif current_color == 'white': 
            color = 'black'
        else: 
            print("Invalid color!")
            color = current_color

        return color



    def _get_wave_func(self, duration, fps):
        """
        Approximate a sine wave so all changes in value
        land on a whole frame given your sampling rate using
        the sampled sinusoidal stimulation method [1]

        References
        ----------

        [1] - Chen, X., Chen, Z., Gao, S., & Gao, X. (2014). 
            A high-itr ssvep-based bci speller. 
            Brain-Computer Interfaces, 1(3-4), 181-191.

        Parameters
        ----------

        duration: float
            Duration of the wave in seconds.

        wave_type: str, ['sin', 'square', AM, FM, FSK, code, mseq]
            default = 'sin'.
            Type of stimuli to present. 'sin' and 'square' are SSVEP.
            'AM', 'FM' and 'FSK' are SSVEP with modulation. 'code' and 
            'mseq' are code-VEP.

        sfreq: int or None, default = None
            Sampling frequency of your wave. If None, it
            is sate as the same value of sampling_rate.

        return_time: bool, default = False
            If True, return the time vector. Useful for plotting 
            the transformed wave.

        Returns
        -------

        wave: np.array
            Array with the approximated sine wave. Values range
            between 0 and 1.
        """

        if self.wave_type == 'sin':
            wave_func = np.sin
        elif self.wave_type == 'square':
            wave_func = scipy.signal.square
        elif self.wave_type=='code':
            A = 3.882
            x = 0.15

        # Since every time point corresponds to one frame, we just need a list from 0 to 
        # duration * fps
        length = int(duration * fps)

        # Construct your wave

        if self.wave_type == 'code':
            """Chaotic codes from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6402685/pdf/pone.0213197.pdf
            """
            wave = []
            for i in range(63):
                x = A*x*(1-x)
                if x>0.5:
                    wave.append(0)
                    wave.append(1)
                else:
                    wave.append(1)
                    wave.append(0)
            wave = np.array(wave)
            shift = int(self.freq * 63 / 10)
            wave = np.roll(wave, shift)
            wave = np.tile(wave, int(np.ceil(length/63)))

        elif self.wave_type == 'mseq':
            """Maximum length sequences https://en.wikipedia.org/wiki/Maximum_length_sequence
            """
            wave,_ = scipy.signal.max_len_seq(4, length=63)
            shift = int(self.freq * 63 / 10)
            wave = np.roll(wave, shift)
            wave = np.tile(wave, int(np.ceil(length/63)))

        elif self.wave_type == 'random':
            """Random sequence composed of subsequences of length 15 with 6 to 8 bit changes.
            As in EEG2Code.
            """
            wave = []
            for _ in range(5):
                seq = np.random.binomial(1,0.5,size=15)
                mean = seq.mean()
                while mean < 0.4 or mean > 0.53: # Between 6 and 8 changes of bits in a sequence of 15 bits
                    seq = np.random.binomial(1,0.5,size=15)
                    mean = seq.mean()
                wave.extend(seq)
            wave = np.array(wave)
            wave = wave[:63]

            wave = np.tile(wave, int(np.ceil(length/63)))
            wave = wave[:length]

        elif self.wave_type == 'random_slow':
            """Random sequence with 1 possible change every 7 bits.
            """
            wave = []
            end = np.random.binomial(1,0.5)
            start = np.abs(1-end)
            seq = np.linspace(start, end, num=6)
            start = end
            wave.extend(seq)
            for _ in range(10):
                end = np.random.binomial(1,0.5)
                seq = np.linspace(start, end, num=6)
                start = end
                wave.extend(seq)
            wave = np.array(wave)
            wave = wave[:63]

            wave = np.tile(wave, int(np.ceil(length/63)))
            wave = wave[:length]

        elif (self.wave_type == 'sin') or (self.wave_type == 'square'):
            frame_index = np.arange(0, length, 1)
            wave = 0.5 * (1 + wave_func(2 * np.pi * self.freq * (frame_index / fps) + (self.phase * np.pi)))
        
        return wave

    def _get_boundaries(self):
        """Calculate the four boundaries of the screen based on the screen resolution"""

        # First, find the boundaries of the stim in the current resolution
        width, height = self.window.size

        # Unpack flicker size
        x_size, y_size = self.total_size

        # Create and populate the dict with the boundaries
        boundary_dict = {}

        boundary_dict["plus_x"] = (width // 2) - (self.size // 2) * self.rows
        boundary_dict["minus_x"] = (-width // 2) + (self.size // 2) * self.rows
        boundary_dict["plus_y"] = (height // 2) - (self.size // 2) * self.cols
        boundary_dict["minus_y"] = (-height // 2) + (self.size // 2) * self.cols

        return boundary_dict

    def _check_base_pos(self, base_pos):
        """
        Check that user's desired position is within boundaries, then return the position.
        """

        if type(base_pos) == tuple:
            # Unpack user-given position
            x_pos, y_pos = base_pos

            # Check the positions and give warnings (if any)
            if x_pos < self.bound_pos["minus_x"] or x_pos > self.bound_pos["plus_x"]:
                warnings.warn(message=f"X position out of bounds. Valid values range from " 
                                    f"{self.bound_pos['minus_x']} and {self.bound_pos['plus_x']}")
            
            if y_pos < self.bound_pos["minus_y"] or y_pos > self.bound_pos["plus_y"]:
                warnings.warn(message=f"Y position out of bounds. Valid values range from " 
                                    f"{self.bound_pos['minus_y']} to {self.bound_pos['plus_y']}")
        elif type(base_pos) == str:
            # Check for x-axis keywords
            if 'left' in base_pos:
                x_pos = self.bound_pos["minus_x"]
            elif 'right' in base_pos:
                x_pos = self.bound_pos["plus_x"]
            else:
                warnings.warn("Absent or invalid X-position string argument. Valid values are "
                              "'left' and 'right. Using 0 as X-position value...")
                x_pos = 0

            if 'down' in base_pos:
                y_pos = self.bound_pos["minus_y"]
            elif 'up' in base_pos:
                y_pos = self.bound_pos["plus_y"]
            else:
                warnings.warn("Absent or invalid Y-position string argument. Valid values are "
                              "'up' and 'down. Using 0 as Y-position value...")
                y_pos = 0

        else:
            warnings.warn("Invalid position argument. It must be a tuple of integers or a "
                          "correctly-formatted string. Using default position (0, 0)...")

            x_pos, y_pos = (0, 0)

        return x_pos, y_pos

    def _make_board(self):
        """
        Make the checkerboard. Create positions around the center based on the
        desired dimensions of the board.
        """

        # Initialize array to populate
        board_array = np.empty((self.rows, self.cols), dtype=object)

        # Starting color
        first_color = 'white'

        # Iterate over rows and columns to create rectangles
        for i in range(self.rows):

            # Color for first square of the raw
            square_color = self._alternate_color(first_color)

            # Store it separatedly
            first_color = square_color

            # Check if number of rows is odd or even to get position offset
            if self.rows % 2 == 0:
                row_offset = 50 + self.base_x
            elif self.rows % 2 == 1:
                row_offset = 0 + self.base_x

            # Same for columns
            if self.cols % 2 == 0:
                col_offset = 50 + self.base_y
            elif self.cols % 2 == 1:
                col_offset = 0 + self.base_y

            x_pos = i * self.size - self.rows // 2 * self.size + row_offset

            for j in range(self.cols):

                y_pos = j * self.size - self.cols // 2 * self.size + col_offset

                # Create the rectangle
                

                rect = visual.Rect(self.window,
                                height = self.height,
                                width = self.width,  
                                pos=(x_pos, y_pos), 
                                lineColor=square_color, fillColor=square_color)


                # Add it to the array
                board_array[i, j] = rect

                square_color = self._alternate_color(square_color)

        return board_array

    def draw(self, time):
        """
        Draw the checkerboard with such contrast in each square so it flickers at
        the desired frequency. Each call to this function corresponds to one frame
        of presentation.

        At each frame, the contrast of each square will receive a different value
        according to a sin wave modulated by the desired frequency.
        
        Parameters
        ----------
        
        time: float
              Time coming from core.Clock in psychopy, determines the frame
        """

        # Get the corresponding value for our desired frequency (between 0 and 1)
        opac_val = 0.5 * (1 + self.wave_type(time * np.pi * 2 * self.freq))

        # Scalate it with the desired amplitude
        opac_val *= self.amp

        for i in range(self.rows):
            for j in range(self.cols):
                self.board[i, j].opacity = opac_val
                self.board[i, j].draw()
    
    def draw2(self, frame, amp_override=None):
        """
        Draw the checkerboard with a pre-calculated wave that only takes into
        account the frame index of the wave to decide on the opacity level.
        """

        # Get opacity for the flickers
        if amp_override:
            opac_val = amp_override
        else:
            opac_val = self.wave_func[frame] * self.amp

        for i in range(self.rows):
            for j in range(self.cols):
                self.board[i, j].opacity = opac_val
                self.board[i, j].draw()


def get_screen_settings(platform):
    """
    Get screen resolution and refresh rate of the monitor

    Parameters
    ----------

    platform: str, ['Linux', 'Ubuntu']
              output of platform.system, determines the OS running this script

    Returns
    -------

    height: int
            Monitor height

    width: int
           Monitor width

    """

    if platform not in ["Linux", "Windows"]:
        print("Unsupported OS! How did you arrive here?")
        sys.exit()

    if platform == "Linux":
        cmd = ["xrandr"]
        cmd2 = ["grep", "*"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(cmd2, stdin=p.stdout, stdout=subprocess.PIPE)

        p.stdout.close()

        info, _ = p2.communicate()
        screen_info = info.decode("utf-8").split()[:2]  # xrandr gives bytes, for some reason

        width, height = list(map(int, screen_info[0].split("x")))  # Convert to Int

    elif platform == "Windows":

        width, height = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)

    return width, height




