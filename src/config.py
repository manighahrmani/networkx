import platform

# Determine the operating system
os_type = platform.system()

if os_type == "Windows":
    SOLVER_PATH = "C:\\Users\\manig\\Downloads\\PACE2017-min-fill"
elif os_type == "Darwin":  # Darwin is the operating system for MacOS
    SOLVER_PATH = "/Users/manigh/Desktop/PACE2017-min-fill"
else:
    raise EnvironmentError("Unsupported operating system")

ROWS = 2
MAX_COLUMNS = 12
