import pandas as pd
import cv2
import numpy as np

INPUT_SHAPE = (66, 200, 3) # in case you change this, it will be necessary to update model.py and drive.py as well.

# Vivek Yadav
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.nikqu6aas
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


# Vivek Yadav
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.nikqu6aas
def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image


# This is a modified version of code created by Mez Gebre
# https://mez.github.io/deep%20learning/2017/02/14/mainsqueeze-the-52-parameter-model-that-drives-in-the-udacity-simulator/
def crop_top_and_bottom(image):
    resized = cv2.resize(image[67:135], (INPUT_SHAPE[1],INPUT_SHAPE[0]),  cv2.INTER_AREA)
    #resized = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)[:,:,1] #in case want to convert to 1 channel
    return resized


# This is a modified version of code created by Mez Gebre
# https://mez.github.io/deep%20learning/2017/02/14/mainsqueeze-the-52-parameter-model-that-drives-in-the-udacity-simulator/
def shift_img(image, random_shift):
    rows, cols = image.shape[0], image.shape[1]
    mat = np.float32([[1, 0, random_shift], [0, 1, 0]])
    return cv2.warpAffine(image, mat, (cols, rows))


def load_image(row):
    """
    This is the main workhorse. It takes a dataframe row and loads the image; making
    proper augmentations based on the flags included. Assumes images are in a
    directory called './data'. Also assumes that the dataframe row has gone through
    the proper modifications in the preprocessing step. Refer to the playground
    notebook for more info.
    Args:
        row: dataframe row
    Returns:
        image: nd.array
    """
    image = cv2.imread("./data/{0}".format(row.image.strip()))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = augment_brightness_camera_images(image)
    image = add_random_shadow(image)
    image = crop_top_and_bottom(image)    
    
    if(row.is_flipped):
        image = cv2.flip(image,1)
    if(row.is_shift): 
        image = shift_img(image, row.random_shift)  
    return image



def get_processed_dataframes():
    """
    Assumes the proper modifications in the preprocessing step. Refer to the playground
    notebook for more info. Assumes the preprocessed csv driver log is titeled
    'preprocessed_driver_log.csv'
    Returns:
        df: dataframe
    """
    return pd.read_csv('preprocessed_driver_log.csv')