#Configuration for units=deg

# HEIGHT: int = 1.49
# WIDTH: int = 1.78
# UNITS = "deg"
# NO_ROWS: int = 1
# NO_COLUMNS: int = 3
# HEIGHT_OF_TARGET = 0.7
# HORI_DIVIDER_START = [-19.12,-2.7]
# HORI_DIVIDER_END = [19.12,-2.7]
# VER_DIVIDER_1_START = [-7.71,10.98]
# VER_DIVIDER_1_END = [-7.71, -10.98]
# VER_DIVIDER_2_START = [7.71,10.98]
# VER_DIVIDER_2_END = [7.71, -10.98]
# DISPLAY_BOX_SIZE = [38.4,3]
# DISPLAY_BOX_POS = (0,13)

#Configuration for units=pix
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
NUM_BLOCK: int = 5
NUM_TRIAL: int = 1
NUM_SESSION: int = 3
EPOCH_DURATION: float = 3
ITI_DURATION: float = 0.1
CUE_DURATION: float = 0.7
NO_SUBSPELLER: int = 6

FREQS: list = [8, 8.2, 8.4, 8.6, 8.8, 9, 9.2, 9.4, 9.6]

POSITIONS: list = [(-800, 300), (0, 300), (800, 300), (-800, 0), (0,0), (800, 0), (-800, -300), (0, -300), (800, -300)]

AMPLITUDE: float = 1.0

PHASES: list = [0 , 0.35 , 0.70 , 1.05 , 1.40 , 1.75, 0.10, 0.45, 0.80 ]

TARGET_CHARACTERS:list = ["A", "B", "C", "J", "K", "L", "S", "T", "U"]

SUBSPELLERS:dict = {"1": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
                    "2": ["J", "K", "L", "M", "N", "O", "P", "Q", "R"],
                    "3": ["S", "T", "U", "V", "W", "X", "Y", "Z", "0"],
                    "4": ["1", "2", "3", "4", "5", "6"],
                    "5": ["7", "8", "9", ".", "?", ","],
                    "6": ["(", "Space", ")", "!", "-", "<<"]}
SERIAL_PORT:str = "COM3"
BOARD_ID:int = 8
PARTICIPANT_ID:str = "buffer_online"
RECORDING_DIR:str = "simple_ssvep_v2/record"
TYPE_OF_FILE:str = ".fif"
CSV_DIR:str = "csv/"
BLOCK_BREAK:int = 1
MARKERS:dict = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0, "G": 7.0, "H": 8.0, "I": 9.0, "J": 10.0, "K": 11.0, "L": 12.0, "M": 13.0, "N": 14.0, "O": 15.0, "P": 16.0, "Q": 17.0, "R": 18.0, "S": 19.0, "T": 20.0, "U": 21.0, "V": 22.0, "W": 23.0, "X": 24.0, "Y": 25.0, "Z": 26.0, "0": 27.0, "1": 28.0, "2": 29.0, "3": 30.0, "4": 31.0, "5": 32.0, "6": 33.0, "7": 34.0, "8": 35.0, "9": 36.0, ".": 37.0, "?": 38.0, ",": 39.0, "(": 40.0, "Space": 41.0, ")": 42.0, "!": 43.0, "-": 44.0, "<<":45.0, "trial_start":99.0}