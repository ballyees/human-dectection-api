# reference: https://github.com/gabrielcassimiro17/raspberry-pi-tensorflow
from chardet import detect
import tensorflow_hub as hub
import tensorflow as tf
import cv2

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
    
    def run(self, image, thresh=0.5):
        image = self.preprocessing(image)
        boxes, scores, classes, num_detections = self.model(image)
        pred_labels = classes.numpy().astype('int')[0]
        human_idx = pred_labels == 1
        pred_scores = scores.numpy()[0]
        pred_scores = pred_scores[human_idx]
        pred_labels = pred_labels[human_idx]
        return (pred_scores, pred_scores[pred_scores >= thresh].shape[0])
    
detector = HumanDetectorModel()