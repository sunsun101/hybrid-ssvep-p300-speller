from psychopy import visual, event
import numpy as np

# Set up the window
window = visual.Window([1920, 1080], screen=1, color=[1,1,1],blendMode='avg', useFBO=True, units="deg", monitor="speller")

# Set up the stimuli
n_rows = 5  # number of rows
n_cols = 9  # number of columns
monitor_width = 60  # monitor width in cm
monitor_height = 33  # monitor height in cm
viewing_distance = 60  # viewing distance in cm
stim_size = [1.78, 1.49]  # size of each stimulus in degrees
gap_size = [4, 4]  # gap size between stimuli in degrees
stim_positions = np.empty((n_rows, n_cols), dtype=object)  # array to hold stimulus positions

# Calculate stimulus positions
x_start = -(n_cols-1)*(stim_size[0]+gap_size[0])/2  # starting x-position
y_start = (n_rows-1)*(stim_size[1]+gap_size[1])/2  # starting y-position
for i in range(n_rows):
    for j in range(n_cols):
        x_pos = x_start + j*(stim_size[0]+gap_size[0])
        y_pos = y_start - i*(stim_size[1]+gap_size[1])
        stim_positions[i,j] = [x_pos, y_pos]

# Create the stimuli
stim_color = 'black'
stim_list = []
for i in range(n_rows):
    for j in range(n_cols):
        stim = visual.Rect(window, size=stim_size, pos=stim_positions[i,j], color=stim_color)
        stim_list.append(stim)

# Draw the stimuli
for stim in stim_list:
    stim.draw()

# Update the window and wait for a response
window.flip()
event.waitKeys()
window.close()
