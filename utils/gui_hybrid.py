"""
Modified from Ludovic Darmet's Checkerboard Code. Credits to him

Author: Ludovic Darmet; Juan Jesus Torre Tresols
Mail: ludovic.darmet@isae-supaero.fr; Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""


import scipy.signal
import sys
import os
import sys
import platform
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


class Stimuli:

    def __init__(self, window, frequency=10, phase=0, amplitude=1., wave_type='sin', duration=2.2, fps=60, base_pos=(0, 0), height=100, width=100):
        self.window = window
        self.freq = frequency
        self.phase = phase
        self.amp = amplitude
        self.wave_type = wave_type
        self.height = height
        self.width = width
        self.base_x, self.base_y = base_pos
        self.wave_func = self._get_wave_func(duration, fps)

        self.board = self._make_stimuli(base_pos)



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

        # Since every time point corresponds to one frame, we just need a list from 0 to 
        # duration * fps
        length = int(duration * fps)

        if (self.wave_type == 'sin') or (self.wave_type == 'square'):
            frame_index = np.arange(0, length, 1)
            wave = 0.5 * (1 + wave_func(2 * np.pi * self.freq * (frame_index / fps) + (self.phase * np.pi)))
        
        return wave

    

    def _make_stimuli(self, base_pos):
        """
        Just draw a rectangle
        """
        x_pos, y_pos = base_pos
        color = 'black'

        rect = visual.Rect(self.window,
                        height = self.height,
                        width = self.width,  
                        pos=(x_pos, y_pos), 
                        lineColor=color, fillColor=color)

        return rect
    
    def draw(self, time):
        opac_val = 0.5 * (1 + self.wave_type(time * np.pi * 2 * self.freq))

        # Scalate it with the desired amplitude
        opac_val *= self.amp
        self.board.opacity = opac_val
        self.board.draw()
    
    def draw2(self, frame, amp_override=None):

        # Get opacity for the flickers
        if amp_override:
            opac_val = amp_override
        else:
            opac_val = self.wave_func[frame] * self.amp

            self.board.opacity = opac_val
            self.board.draw()

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




