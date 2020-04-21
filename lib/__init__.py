import os
import sys

lib_path = os.path.join(os.path.dirname(__file__))
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
