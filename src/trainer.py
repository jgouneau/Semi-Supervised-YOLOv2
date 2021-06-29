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
    
    agent_config_path = main_config['agents_config_path']
    with open(agent_config_path) as config_buffer:    
        agent_config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations 
    ###############################

    if dataset_config['annotation_type'] == 'xml':
        # parse annotations of the training set
        train_imgs, _ = parse_annotation_xml(dataset_path + dataset_config['train']['ann_folder'], 
                                                        dataset_path + dataset_config['train']['img_folder'],
                                                        dataset_config['labels'])

        # parse annotations of the validation set, if any.
        if os.path.exists(dataset_config['valid']['ann_folder']):
            valid_imgs, _ = parse_annotation_xml(dataset_path + dataset_config['valid']['ann_folder'], 
                                                            dataset_path + dataset_config['valid']['img_folder'],
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

    #########################################
    #   Load the pretrained weights (if any) 
    #########################################

    # TODO : adapter cette partie
    #if os.path.exists(agent_config['train']['pretrained_weights']):
    #    print("Loading pre-trained weights in", agent_config['train']['pretrained_weights'])
    #    agent.load_weights(agent_config['train']['pretrained_weights'])

    ###############################
    #   Start the training process 
    ###############################

    agent.train(train_data=train_imgs,
               valid_data=valid_imgs,
               nb_epochs=agent_config['train']['nb_epochs'],
               batch_size=agent_config['train']['batch_size'],
               learning_rate=agent_config['train']['learning_rate'],
               # TODO : implémenter le warmup
               #warmup_epochs=config['train']['warmup_epochs'],
               lamb_obj = agent_config['train']['lamb_obj'],
               lamb_noobj = agent_config['train']['lamb_noobj'],
               lamb_coord = agent_config['train']['lamb_coord'],
               lamb_class = agent_config['train']['lamb_class'],
               workers=agent_config['train']['workers'],
               max_queue_size=agent_config['train']['max_queue_size'])