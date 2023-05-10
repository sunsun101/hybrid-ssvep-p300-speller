#Configuration for units=pix
HEIGHT: int = 140
WIDTH: int = 140
UNITS = "pix"
HEIGHT_OF_TARGET = 35
HORI_DIVIDER_START = [-900,0]
HORI_DIVIDER_END = [900,0]
VER_DIVIDER_1_START = [0,500]
ONLINE_VER_DIVIDER_1_START = [0, 400]
VER_DIVIDER_1_END = [0,-500]
DISPLAY_BOX_SIZE = [1700,100]
DISPLAY_BOX_POS = (0,450)

SIZE: int = 100
NUM_BLOCK: int = 15
NUM_TRIAL: int = 3
NUM_SESSION: int = 3
EPOCH_DURATION: float = 3
ITI_DURATION: float = 0.1
CUE_DURATION: float = 1
NO_SUBSPELLER: int = 4

FREQS: list = [8, 8, 8.6, 8.6, 9, 9, 9.6, 9.6 ]

POSITIONS: list = [(-800, 200), (- 400, 200), (400, 200), (800, 200), (-800,-200), (-400, -200), (400, -200), (800, -200)]

AMPLITUDE: float = 1.0

PHASES: list = [0 , 0 , 1.05 , 1.05 , 1.75 , 1.75, 0.80, 0.80]

TARGET_CHARACTERS:list = ["A", "B", "C", "D", "E", "F", "G", "H"]

SUBSPELLERS:dict = {"1": ["A", "B"],
                    "2": ["C", "D"],
                    "3": ["E", "F"],
                    "4": ["G", "H"]
                    }

SERIAL_PORT:str = "COM3"
BOARD_ID:int = 8
PARTICIPANT_ID:str = "amanda_20230510"
RECORDING_DIR:str = "hybrid/record"
TYPE_OF_FILE:str = ".fif"
CSV_DIR:str = "csv/"
BLOCK_BREAK:int = 30
MARKERS:dict = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0, "G": 7.0, "H": 8.0, "I": 9.0, "trial_start": 99.0}