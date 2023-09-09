HEIGHT: int = 100
WIDTH: int = 100
UNITS = "pix"
HEIGHT_OF_TARGET = 35
HORI_DIVIDER_START = [-850,-75]
HORI_DIVIDER_END = [850,-75]
VER_DIVIDER_1_START = [-300,350]
VER_DIVIDER_1_END = [-300,-350]
VER_DIVIDER_2_START = [300, 350]
VER_DIVIDER_2_END = [300,-350]
DISPLAY_BOX_SIZE = [1700,100]
DISPLAY_BOX_POS = (0,450)

SIZE: int = 100
NUM_BLOCK: int = 1
NUM_TRIAL: int = 5
NUM_SESSION: int = 3
EPOCH_DURATION: float = 5
ITI_DURATION: float = 0.1
CUE_DURATION: float = 0.5
# FREQS: list = [8,9,10,11,12,13,14,15,16]
FREQS: list = [8, 8.2, 8.4, 8.6, 8.8, 9, 9.2, 9.4, 9.6]

POSITIONS: list = [(-800, 300), (0, 300), (800, 300), (-800, 0), (0,0), (800, 0), (-800, -300), (0, -300), (800, -300)]

AMPLITUDE: float = 1.0

# PHASES:list = [0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,0]
PHASES: list = [0 , 0.35 , 0.70 , 1.05 , 1.40 , 1.75, 0.10, 0.45, 0.80 ]

TARGET_CHARACTERS:list = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

SERIAL_PORT:str = "COM3"
BOARD_ID:int = 8
PARTICIPANT_ID:str = "sunsun_20230331_online"
RECORDING_DIR:str = "wang_experiment/record"
TYPE_OF_FILE:str = ".fif"
CSV_DIR:str = "csv/"
BLOCK_BREAK:int = 1
MARKERS:dict = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0, "G": 7.0, "H": 8.0, "I": 9.0, "J": 10.0, "K": 11.0, "L": 12.0, "M": 13.0, "N": 14.0, "O": 15.0, "P": 16.0, "Q": 17.0, "R": 18.0, "S": 19.0, "T": 20.0, "U": 21.0, "V": 22.0, "W": 23.0, "X": 24.0, "Y": 25.0, "Z": 26.0, "0": 27.0, "1": 28.0, "2": 29.0, "3": 30.0, "4": 31.0, "5": 32.0, "6": 33.0, "7": 34.0, "8": 35.0, "9": 36.0, " ": 37.0, ",": 38.0, ".": 39.0, "<": 40.0}