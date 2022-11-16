from turtle import fillcolor, pos
from psychopy import visual, core, event #import some libraries from PsychoPy
import platform
import os
from utils_experiments import get_screen_settings, CheckerBoard
import argparse
import json
import numpy as np
import random

#Load config
path = os.getcwd() + '\presentation'
parser = argparse.ArgumentParser(description='Config file name')
parser.add_argument('-f', '--file', metavar='ConfigFile', type=str,
                    default='speller_config.json', help="Name of the config file for freq "
                                      "and amplitude. Default: %(default)s.")
args = parser.parse_args()
config_path = os.path.join(path, args.file)

with open(config_path, 'r') as config_file:
    params = json.load(config_file)

# Experimental params
size = params['size']
trial_n = params['trial_n']
cal_n = params['cal_n']
epoch_duration = params['epoch_duration']
iti_duration = params['iti_duration']
cue_duration = params['cue_duration']
freq = params['freqs']
positions = [tuple(position) for position in params['positions']]
phases = params['phases']
target_characters = params['target_characters']
amp = params['amplitude']


# Window parameters
system = platform.system()
width, height = get_screen_settings(system)

#create a window
window = visual.Window([width, height], screen=1, color=[1,1,1],blendMode='avg', useFBO=True, units="pix", fullscr=True)
refresh_rate = round(window.getActualFrameRate())
print("Refresh Rate ==>", refresh_rate)

# Time conversion to frames
epoch_frames = int(epoch_duration * refresh_rate)
print("Epoch frames ==>",epoch_frames)
iti_frames = int(iti_duration * refresh_rate)
iti_frames_cal = int(0.8 * refresh_rate)
cue_frames = int(cue_duration * refresh_rate)

#Presentation content

cue = visual.Rect(window, width=100, height=80, pos=[0, 0], lineWidth=3, lineColor='red')

calib_text_start = "Starting callibration phase.Please avoid moving or blinking.\n\
You may blink when shifting your gaze.Focus your target on the characters presented with red cue."

calib_text_end = "Calibration phase completed"
cal_start = visual.TextStim(window, text=calib_text_start, color=(-1., -1., -1.))
cal_end = visual.TextStim(window, text=calib_text_end, color=(-1., -1., -1.))

targets = {f"{target}": visual.TextStim(win=window, text=target, pos=pos, color=(-1., -1., -1.), height=35)
           for pos, target in zip(positions, target_characters)}


wave_type = "sin"
flickers = {f"{target}": CheckerBoard(window=window, size=size, frequency=f, phase=phase, amplitude=amp, 
                                    wave_type=wave_type, duration=epoch_duration, fps=refresh_rate,
                                    base_pos=pos)
            for f, pos, phase, target in zip(freq, positions, phases, target_characters)}

hori_divider = visual.Line(window, start=[-850,-75], end=[850,-75], lineColor='black')
ver_divider_1 = visual.Line(window, start=[-300,350], end=[-300,-350], lineColor='black')
ver_divider_2 = visual.Line(window, start=[300, 350], end=[300,-350], lineColor='black')
display_box = visual.Rect(window, size=[1700,100], pos=(0,450), lineColor='black', lineWidth=2.5)

trial_list = []


for _ in range(cal_n):
    sequence = random.sample(target_characters, 45)
    trial_list.append(sequence)

# Starting the display
trialClock = core.Clock()
cal_start.draw()
window.flip()
# core.wait(6)

for idx, sequence in enumerate(trial_list):
    # Drawing display box
    display_box.autoDraw = True
    # Drawing the grid
    hori_divider.autoDraw = True
    ver_divider_1.autoDraw = True
    ver_divider_2.autoDraw = True
    # Display target characters
    for target in targets.values():
            target.autoDraw = True

    # For the flickering
    for target in sequence:

        target_flicker = flickers[str(target)]
        target_pos = (target_flicker.base_x, target_flicker.base_y)
        target_freq = target_flicker.freq
        target_phase = target_flicker.phase

        # for n in range(iti_frames):
        #     for flicker in flickers.values():
        #         flicker.draw2(frame=0, amp_override=1.)
        #     window.flip()

        t0 = trialClock.getTime()  # Retrieve time at start of cue presentation
        #Display the cue
        cue.pos = target_pos
        for frame in range(cue_frames):
                # for flicker in flickers.values():
                #     flicker.draw2(frame=0, amp_override=1.)

                # Draw the cue over the static flickers
                cue.draw()
                window.flip()

        frames = 0

        for frame, n in enumerate(range(epoch_frames)):
            for flicker in flickers.values():
                flicker.draw2(frame=frame)
            frames += 1
            window.flip()

        # At the end of the trial, calculate real duration and amount of frames
        t1 = trialClock.getTime()  # Time at end of trial
        elapsed = t1 - t0
        print(f"Time elapsed: {elapsed}")
        print(f"Total frames: {frames}")
        if len(event.getKeys())>0:
            break
        event.clearEvents()



#cleanup
window.close()
core.quit()