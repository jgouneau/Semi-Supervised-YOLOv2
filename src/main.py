from numpy.core.arrayprint import _make_options_dict
from manager import hire_agent
from trainer import train_agent
from predictor import agent_prediction
from evaluator import evaluate_agent

MAIN_CONFIG_PATH = "./YOLOv2/config.json"

def _main_():
    agent = hire_agent('zoologist', 'raccoon', MAIN_CONFIG_PATH)
    valid_data = train_agent(agent, 'raccoon', MAIN_CONFIG_PATH)
    evaluate_agent(agent, 'raccoon', valid_data, MAIN_CONFIG_PATH)
    print("helloooo")

if __name__ == '__main__':
    _main_()