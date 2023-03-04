import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.speller_config import * 

BoardShim.enable_dev_board_logger()

params = BrainFlowInputParams()
params.serial_port = SERIAL_PORT

board = BoardShim(BOARD_ID, params)
board.prepare_session()

sampling_rate = 250
num_channels = 8


board.start_stream(45000, 'file://test.csv:w')
data = np.zeros((num_channels, sampling_rate * 5))

plt.ion()
fig, ax = plt.subplots(num_channels, figsize=(15,10))

while True:
    board_data = board.get_current_board_data(num_channels * sampling_rate)

    data[:, :-sampling_rate] = data[:, sampling_rate:]
    data[:, -sampling_rate:] = board_data

    for i in range(num_channels):
        ax[i].clear()
        ax[i].plot(data[i])

    plt.draw()
    plt.pause(0.001)







