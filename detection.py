import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

# Formula from numpy form image to change from rgb to gray scale
def rgbToGray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
        
    """
    # Begin your code (Part 4)
    
    # Read lines from txt file
    with open(dataPath,'r') as f:
        txtLines = f.readlines()
        
    path = "data/detect"
    l = 0
    
    while l < len(txtLines):
        fileName, people = txtLines[l].split(" ")
        people = int(people)
        image = mpimg.imread(os.path.join(path, fileName))
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for n in range(1, people + 1):
            
            # Get part of the image which was chosen by the txt file
            x, y, width, height = txtLines[l + n].split(" ")
            x = int(x)
            y = int(y)
            width = int(width)
            height = int(height)
            
            # make it a small image
            imageCrop = image[y:y+height, x:x+width, :]
            
            # Apply gray scale
            imageCrop = rgbToGray(imageCrop)
            
            # Resize the small image
            imageTran = Image.fromarray(imageCrop)
            imageTran = imageTran.resize((19, 19))
            imageCrop = np.array(imageTran)
            
            # Apply our classifier
            result = clf.classify(imageCrop)
            
            if result == 1:
                rect = plt.Rectangle((x, y), width, height, fill=False, edgecolor = 'green',linewidth=1)
            else:
                rect = plt.Rectangle((x, y), width, height, fill=False, edgecolor = 'red',linewidth=1)
            
            # Show the result on the image
            ax.add_patch(rect)
            plt.imshow(image)
           
        plt.show()
        l += people + 1
    
    # End your code (Part 4)
