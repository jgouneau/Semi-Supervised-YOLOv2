import json
import os

import numpy as np

from core.yolo import YOLO
from core.utils import enable_memory_growth, parse_annotation_xml

def train_agent(agent, dataset_name, main_config_path):
    enable_memory_growth()

    with open(main_config_path) as config_buffer:    
        main_config = json.loads(config_buffer.read())

    dataset_path = main_config['datasets_paths'][dataset_name]
    with open(dataset_path + "config.json") as config_buffer:    
        dataset_config = json.loads(config_buffer.read())
    
    agent_path = main_config['agents_paths'][agent.name]
    with open(agent_path + "config.json") as config_buffer:    
        agent_config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations 
    ###############################
    train_mode = agent_config['train']['mode']

    if dataset_config['learning_type'] == "supervised":
        train_folder = dataset_config['train']
        unlab_train_folder = 0
    else:
        train_folder = dataset_config['train']['labelled']
        unlab_train_folder = dataset_config['train']['unlabelled']
    test_folder = dataset_config['test']

    if dataset_config['annotation_type'] == 'xml':
        # parse annotations of the training set
        train_imgs = parse_annotation_xml(dataset_path + train_folder['ann_folder'],
                                          dataset_path + train_folder['img_folder'],
                                          dataset_config['labels'])
        if train_mode == "semi-supervised":
            if unlab_train_folder != 0:
                unlab_train_imgs = parse_annotation_xml(dataset_path + unlab_train_folder['ann_folder'], 
                                                        dataset_path + unlab_train_folder['img_folder'],
                                                        dataset_config['labels'])
            else:
                raise ValueError(
                    "no unlabelled folder for semi-supervised learning")

        # parse annotations of the validation set, if any.
        if dataset_config['test']['ann_folder'] != "":
            valid_imgs = parse_annotation_xml(dataset_path + test_folder['ann_folder'], 
                                              dataset_path + test_folder['img_folder'],
                                              dataset_config['labels'])
            split = False
        else:
            split = True
    else:
        raise ValueError(
            "'annotations_type' must be 'xml' not {}.".format(dataset_config['annotations_type']))

    if split:
        train_valid_split = int(0.8 * len(train_imgs))
        np.random.shuffle(train_imgs)

        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]

    ###############################
    #   Start the training process 
    ###############################
    if train_mode == 'supervised':
        agent.train(train_data = train_imgs,
                  valid_data = valid_imgs,
                  nb_epochs = agent_config['train']['nb_epochs'],
                  batch_size = agent_config['train']['batch_size'],
                  learning_rate = agent_config['train']['learning_rate'],
                  warmup_epochs = agent_config['train']['warmup_epochs'],
                  lamb_obj = agent_config['train']['lamb_obj'],
                  lamb_noobj = agent_config['train']['lamb_noobj'],
                  lamb_coord = agent_config['train']['lamb_coord'],
                  lamb_class = agent_config['train']['lamb_class'],
                  lamb_u = agent_config['train']['lamb_u'],
                  workers=agent_config['train']['workers'],
                  max_queue_size=agent_config['train']['max_queue_size'])
    else :
        agent.train(train_data = train_imgs,
                  valid_data = valid_imgs,
                  nb_epochs = agent_config['train']['nb_epochs'],
                  batch_size = agent_config['train']['batch_size'],
                  learning_rate = agent_config['train']['learning_rate'],
                  warmup_epochs = agent_config['train']['warmup_epochs'],
                  lamb_obj = agent_config['train']['lamb_obj'],
                  lamb_noobj = agent_config['train']['lamb_noobj'],
                  lamb_coord = agent_config['train']['lamb_coord'],
                  lamb_class = agent_config['train']['lamb_class'],
                  lamb_u = agent_config['train']['lamb_u'],
                  pseudo_lab_data=unlab_train_imgs,
                  pseudo_lab_batch_size=agent_config['train']['pseudo_lab_batch_size'],
                  workers=agent_config['train']['workers'],
                  max_queue_size=agent_config['train']['max_queue_size'])
    
    return valid_imgs