"""
config.py

This module determines the operating system and sets the path to the solver accordingly.
It also sets the number of rows and maximum number of columns for the grid.

The module uses the platform module to determine the operating system. 
If the operating system is Windows, the solver path is set to a specific location on the C drive.
If the operating system is macOS, the solver path is set to a specific location on the desktop.
If the operating system is neither Windows nor Darwin, an EnvironmentError is raised.

The number of rows and the maximum number of columns for the grid are set to 5 and 15, respectively.

Variables:
- os_type: The type of the operating system.
- SOLVER_PATH: The path to the solver.
- ROWS: The number of rows in the grid.
- MAX_COLUMNS: The maximum number of columns in the grid.
"""

import platform

# Determine the operating system
os_type = platform.system()

if os_type == "Windows":
    SOLVER_PATH = "C:\\Users\\manig\\Downloads\\PACE2017-min-fill"
elif os_type == "Darwin":  # Darwin is the operating system for MacOS
    SOLVER_PATH = "/Users/manigh/Desktop/PACE2017-min-fill"
else:
    raise EnvironmentError("Unsupported operating system")

ROWS = 5
MAX_COLUMNS = 16
MIN_COLUMNS = 12

CSV_FILENAME = f'{ROWS}_grid_data.csv'
