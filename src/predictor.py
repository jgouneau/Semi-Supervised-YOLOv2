from core.utils import draw_boxes, enable_memory_growth, list_images
from core.yolo import YOLO
from tqdm import tqdm
from google.colab.patches import cv2_imshow
import numpy as np
import json
import cv2
import os

def agent_prediction(agent, image_path, main_config_path):
    
    enable_memory_growth()

    with open(main_config_path) as config_buffer:
        main_config = json.load(config_buffer)
    
    agent_config_path = main_config['agents_config_path']
    with open(agent_config_path) as config_buffer:
        agent_config = json.load(config_buffer)

    ###########################
    #   Predict bounding boxes
    ###########################

    #if os.path.isfile(image_path):
    image = cv2.imread(image_path)
    boxes = agent.predict(image,
                            iou_threshold=agent_config['predict']['iou_threshold'],
                            score_threshold=agent_config['predict']['score_threshold'])
    image = draw_boxes(image, boxes, agent.labels)
    cv2_imshow(image)
        #cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)
    #else:
    #    detected_images_path = os.path.join(image_path, "detected")
    #    if not os.path.exists(detected_images_path):
    #        os.mkdir(detected_images_path)
    #    images = list(list_images(image_path))
    #    for fname in tqdm(images):
    #        image = cv2.imread(fname)
    #        boxes = agent.predict(image)
    #        image = draw_boxes(image, boxes, config['model']['labels'])
    #        #fname = os.path.basename(fname)
    #        #cv2.imwrite(os.path.join(image_path, "detected", fname), image)