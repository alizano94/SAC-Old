import numpy as np

from tensorflow.keras.preprocessing import image

class Helpers():
    def __init__(self):
        pass

    def preProcessImg(self,img_path,IMG_H=212,IMG_W=212):
        '''
        A function that preprocess an image to fit 
        the CNN input.
        args:
        	-img_path: path to get image
        	-IMG_H: image height
            -IMG_W: image width
        Returns:
            -numpy object containing:
                (dum,img_H,img_W,Chanell)
        '''
        #Load image as GS with st size
        img = image.load_img(img_path,color_mode='grayscale',target_size=(IMG_H, IMG_W))
        #save image to array (H,W,C)
        img_array = image.img_to_array(img)
        
        #Create a batch of images
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch