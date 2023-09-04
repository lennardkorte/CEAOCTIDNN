
import sys
import os

"""Script to run other scripts within docker.
It should not be necessary to change anything here.
Usage is described in the README.
"""

if __name__=="__main__":

    if len(sys.argv) == 1:
        print("No scripts to run.")
    
    for arg in sys.argv[1:]:
        print(f"Now executing script: {arg}")
        os.system(f"python -u {arg}")