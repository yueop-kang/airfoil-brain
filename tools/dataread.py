import os
import numpy as np
import matplotlib.pyplot as plt

def load_airfoil(dir_path):
    airfoils={}
    for (root, directories, files) in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            airfoils[file.split('.')[0]]=np.loadtxt(file_path)
    return airfoils

    