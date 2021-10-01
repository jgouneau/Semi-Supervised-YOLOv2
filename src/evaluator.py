from core.preprocessing import BatchGenerator
from core.utils import enable_memory_growth, parse_annotation_xml
from core.yolo import YOLO
from core.evaluation import MapEvaluation
import json
import os

def evaluate_agent(agent, dataset_name, default_valid_imgs, main_config_path):
    
    enable_memory_growth()

    with open(main_config_path) as config_buffer:    
        main_config = json.loads(config_buffer.read())

    dataset_path = main_config['datasets_paths'][dataset_name]
    with open(dataset_path + "config.json") as config_buffer:    
        dataset_config = json.loads(config_buffer.read())
    
    agent_path = main_config['agents_paths'][agent.name]
    with open(agent_path + "config.json") as config_buffer:    
        agent_config = json.loads(config_buffer.read())

    ##########################
    #   Parse the annotations 
    ##########################
    if dataset_config['learning_type'] == "supervised":
        train_folder = dataset_config['train']
    else:
        train_folder = dataset_config['train']['labelled']
    test_folder = dataset_config['test']

    if dataset_config['annotation_type'] == 'xml':
        # parse annotations of the training set
        train_imgs = parse_annotation_xml(dataset_path + train_folder['ann_folder'],
                                          dataset_path + train_folder['img_folder'],
                                          dataset_config['labels'])

        # parse annotations of the validation set, if any.
        if dataset_config['test']['ann_folder'] != "":
            valid_imgs = parse_annotation_xml(dataset_path + test_folder['ann_folder'], 
                                              dataset_path + test_folder['img_folder'],
                                              dataset_config['labels'])
        else:
            valid_imgs = default_valid_imgs
    else:
        raise ValueError("'annotations_type' must be 'xml' not {}.".format(dataset_config['annotations_type']))

    # remove samples without objects in the image
    for i in range(len(train_imgs)-1, 0, -1):
        if len(train_imgs[i]['object']) == 0:
            del train_imgs[i]

    #########################
    #   Evaluate the network
    #########################

    iou_threshold = agent_config['evaluate']['iou_threshold']
    nms_threshold = agent_config['predict']['nms_threshold']
    obj_threshold = agent_config['predict']['obj_threshold']
    print("calculing mAP for iou threshold = {}".format(iou_threshold))
    generator_config = {
                'IMAGE_H': agent._input_size[0],
                'IMAGE_W': agent._input_size[1],
                'IMAGE_C': agent._input_size[2],
                'GRID_H': agent._grid_h,
                'GRID_W': agent._grid_w,
                'BOX': agent._nb_box,
                'LABELS': agent.labels,
                'CLASS': len(agent.labels),
                'ANCHORS': agent._anchors,
                'BATCH_SIZE': 4,
                'TRUE_BOX_BUFFER': 10 # yolo._max_box_per_image,
            }
            
    valid_generator = BatchGenerator(valid_imgs,
                                      generator_config,
                                      jitter=False)
    valid_eval = MapEvaluation(agent, valid_generator,
                                iou_threshold=iou_threshold,
                                nms_threshold=nms_threshold,
                                obj_threshold=obj_threshold)

    _map, average_precisions = valid_eval.evaluate_map()
    for label, average_precision in average_precisions.items():
        print(agent.labels[label], '{:.4f}'.format(average_precision))
    print('validation dataset mAP: {:.4f}\n'.format(_map))

    train_generator = BatchGenerator(train_imgs, 
                                     generator_config,
                                     jitter=False)  
    train_eval = MapEvaluation(agent, train_generator,
                               iou_threshold=iou_threshold,
                               nms_threshold=nms_threshold,
                               obj_threshold=obj_threshold)

    _map, average_precisions = train_eval.evaluate_map()
    for label, average_precision in average_precisions.items():
        print(agent.labels[label], '{:.4f}'.format(average_precision))
    print('training dataset mAP: {:.4f}'.format(_map))