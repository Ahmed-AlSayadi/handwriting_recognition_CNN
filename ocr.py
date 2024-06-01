import numpy as np
import cv2
from keras.models import model_from_json
import shutil
from keras import backend as K
import os

from HANDWRITING_RECOBNITION import wordSegmentation, prepareImg, preprocess, correction_list, decode_label, decode_batch, get_paths_and_texts, predict_image
state = False
def predict(w_model_predict, test_img):
    res = []
    text = []
    img = prepareImg(cv2.imread(test_img), 64)
    res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    for (j, w) in enumerate(res):
        (wordBox, wordImg) = w
        cv2.imwrite('tmp/%d.png'%j, wordImg)
    imgFiles = os.listdir('tmp')
    imgFiles = sorted(imgFiles)
    for f in imgFiles:
        text.append(predict_image(w_model_predict, 'tmp/'+f, is_word=True))
    shutil.rmtree('tmp')
    text = correction_list(text)
    text1 = ' '.join(text)
    return text1

if __name__=='__main__':
    with open('Resource/model_prediction.json', 'r') as f:
        w_model_predict = model_from_json(f.read())
    w_model_predict.load_weights('Resource/iam_words--17--1.887.h5')
    text1 = predict(w_model_predict, 'Resource/test/5.png')
    print('--------------PREDICT---------------')
    print('[Word model]: ', text1)

