import os 
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataprocessing import bigvul
from dataprocessing import get_dep_add_lines_bigvul

def prepared():
    bigvul()
    get_dep_add_lines_bigvul()
    return 

if __name__ == "__main__":
    prepared()  