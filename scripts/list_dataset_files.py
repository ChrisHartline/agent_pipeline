import os
import sys
root = sys.argv[1]
for dirpath, dirnames, filenames in os.walk(root):
    for f in filenames:
        path = os.path.join(dirpath, f)
        print(path, os.path.getsize(path))
