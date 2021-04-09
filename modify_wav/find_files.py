import os
import shutil

def find_files(filepath, file_ext):
    for (path, dir, files) in os.walk(filepath):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == file_ext:
                print("%s\\%s" % (path, filename))
