import numpy as np
import scipy as sp
import cv2
from keras.preprocessing import image
from keras.models import load_model

classifier = load_model('classifier1.h5')

def get_img_array(path):
    img = image.load_img(path)
    return image.img_to_array(img)


def predict_part(inp_arr_image):
    test_image = cv2.resize(inp_arr_image, (64,64))
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    #training_set.class_indices
    if result[0][0] == 1:
        prediction = 'notfire'
        return False
    else:
        prediction = 'fire'
        return True


def get_cells_img(np_arr_img, n=64): # considers all n X n grids
    sub_imgs = []
    for row in range((np_arr_img.shape[0]//n)+1):
        for col in range((np_arr_img.shape[1]//n)+1):
            c_0 = col*n
            c_1 = min((c_0+n), np_arr_img.shape[1])
            r_0 = row*n
            r_1 = min((r_0+n), np_arr_img.shape[0])
    #         print(c_0, c_1, " | ", r_0, r_1)
            sub_imgs.append(np_arr_img[r_0:r_1, c_0:c_1,: ])
    return sub_imgs


def predict(img_path):
    inp_img = get_img_array(img_path)
    inp_img = cv2.resize(inp_img, (750, 500) )
    fire_pred = [predict_part(img) for img in get_cells_img(inp_img, n=128)]

    fire_cnt = 0
    for p in fire_pred:
        if p:
            fire_cnt += 1
    no_cnt = len(fire_pred) - fire_cnt

    if fire_cnt > 5:
        return True
    else:
        return False



