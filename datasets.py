import os
import matplotlib.image as mpimg
import numpy as np

def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    
    # A list of tuple we need
    listOfImageTuples = []
    for subFolder in os.listdir(dataPath):
        # cd into the folder
        for imagesFileName in os.listdir(os.path.join(dataPath ,subFolder)):
            img = mpimg.imread(os.path.join(dataPath, subFolder, imagesFileName))
            if subFolder == "face" :
                tup = (img ,1) # We make 1 represent face
                listOfImageTuples.append(tup)
            else :
                tup = (img ,0) # We make 0 represent non-face
                listOfImageTuples.append(tup)

    return listOfImageTuples