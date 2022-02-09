# reference: https://github.com/gabrielcassimiro17/raspberry-pi-tensorflow

import tensorflow_hub as hub
import tensorflow as tf
import cv2
import numpy as np

class HumanDetectorModel:
    def __init__(self, path, h=512, w=512):
        self.path = path
        self.model = hub.load(path)
        self.h = h
        self.w = w
    def preprocessing(self, image, isRGB=True):
        image = cv2.resize(image, (self.w, self.h))
        if isRGB is False:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        image = tf.expand_dims(image , 0)
        return image
    def run_count(self, image, isRGB=True, thresh=0.2):
        image = self.preprocessing(image, isRGB)
        _, scores, classes, _ = self.model(image)
        pred_labels = classes.numpy().astype('int')[0]
        pred_scores = scores.numpy()[0]
        isHuman = pred_labels == 1
        pred_scores = pred_scores[isHuman]
        isHuman = pred_scores >= thresh
        return isHuman.sum()
    
    def run(self, image, isRGB=True, thresh=0.2):
        image = self.preprocessing(image, isRGB)
        boxes, scores, classes, _ = self.model(image)
        pred_labels = classes.numpy().astype('int')[0]
        pred_boxes = boxes.numpy()[0].astype('int')
        pred_scores = scores.numpy()[0]
        isHuman = pred_labels == 1
        pred_scores = pred_scores[isHuman]
        pred_labels = pred_labels[isHuman]
        pred_boxes = pred_boxes[isHuman]
        isHuman = pred_scores >= thresh
        pred_scores = pred_scores[isHuman]
        pred_labels = pred_labels[isHuman]
        pred_boxes = pred_boxes[isHuman]
        img = np.squeeze(np.copy(image))
        for score, (ymin,xmin,ymax,xmax) in zip(pred_scores, pred_boxes):
            score_txt = f'{100 * score:.2f}'
            img_boxes = cv2.rectangle(img,(xmin, ymax),(xmax, ymin),(0,255,0),1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_boxes,"person",(xmin, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
            cv2.putText(img_boxes,score_txt,(xmax, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
        return img
detector = HumanDetectorModel('model_detector/')