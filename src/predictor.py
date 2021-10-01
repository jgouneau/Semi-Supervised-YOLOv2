from core.utils import draw_boxes, enable_memory_growth, list_images
from core.yolo import YOLO
from tqdm import tqdm
import numpy as np
import json
import cv2
import os

def agent_prediction(agent, image_path, main_config_path):
    
    enable_memory_growth()

    with open(main_config_path) as config_buffer:
        main_config = json.load(config_buffer)
    
    agent_path = main_config['agents_paths'][agent.name]
    with open(agent_path) as config_buffer:
        agent_config = json.load(config_buffer)

    ###########################
    #   Predict bounding boxes
    ###########################

    #if os.path.isfile(image_path):
    image = cv2.imread(image_path)
    boxes = agent.predict(image,
                            obj_threshold=agent_config['predict']['obj_threshold'],
                            nms_threshold=agent_config['predict']['nms_threshold'])
    image = draw_boxes(image, boxes, agent.labels)
    cv2.imshow(image)
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