import json

from core.yolo import YOLO

def hire_agent(agent_name, dataset_name, main_config_path):

    with open(main_config_path) as config_buffer:
        main_config = json.loads(config_buffer.read())

    dataset_path = main_config['datasets_paths'][dataset_name]
    with open(dataset_path + "config.json") as config_buffer:
        dataset_config = json.loads(config_buffer.read())

    agent_config_path = main_config['agents_config_path']
    with open(agent_config_path) as config_buffer:
        agent_config = json.loads(config_buffer.read())
    
    backend = agent_config['model']['backend']
    input_size_w = agent_config['model']['input_size_w']
    input_size_h = agent_config['model']['input_size_h']
    input_size = (input_size_w, input_size_h)
    
    labels = dataset_config['labels']
    anchors = dataset_config['anchors']

    agent = YOLO(agent_name, backend, input_size, labels, anchors, main_config)
    return agent