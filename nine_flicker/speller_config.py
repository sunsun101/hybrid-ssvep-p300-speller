#Configuration for units=pix
HEIGHT: int = 200
WIDTH: int = 200
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
NUM_TRIAL: int = 3
NUM_SESSION: int = 3
EPOCH_DURATION: float = 5
ITI_DURATION: float = 0.1
CUE_DURATION: float = 0.5
NO_SUBSPELLER: int = 6

FREQS: list = [12, 12.2, 12.4, 12.6, 12.8, 13, 13.2, 13.4, 13.6]

POSITIONS: list = [(-800, 300), (0, 300), (800, 300), (-800, 0), (0,0), (800, 0), (-800, -300), (0, -300), (800, -300)]

AMPLITUDE: float = 1.0

PHASES: list = [0 , 0.35 , 0.70 , 1.05 , 1.40 , 1.75, 0.10, 0.45, 0.80 ]

TARGET_CHARACTERS:list = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

SERIAL_PORT:str = "COM3"
BOARD_ID:int = 8
PARTICIPANT_ID:str = "farheen_20230426"
RECORDING_DIR:str = "nine_flicker/record"
TYPE_OF_FILE:str = ".fif"
CSV_DIR:str = "csv/"
BLOCK_BREAK:int = 20
MARKERS:dict = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0, "G": 7.0, "H": 8.0, "I": 9.0}