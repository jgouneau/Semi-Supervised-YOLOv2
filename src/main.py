from numpy.core.arrayprint import _make_options_dict
from manager import hire_agent
from trainer import train_agent
from pseudolabels_generator import agent_gen_pseudo_labels
from predictor import agent_prediction
from evaluator import evaluate_agent

MAIN_CONFIG_PATH = "./config.json"

def _main_():

    # initialize and train teacher
    teacher = hire_agent('teacher', 'wildlife', MAIN_CONFIG_PATH)
    valid_data = train_agent(teacher, 'wildlife', MAIN_CONFIG_PATH)

    # generate pseudo-labels
    agent_gen_pseudo_labels(teacher, 'wildlife', 0.8, MAIN_CONFIG_PATH)

    #initialize and train student
    student = hire_agent('student', 'wildlife', MAIN_CONFIG_PATH)
    valid_data2 = train_agent(student, 'wildlife', MAIN_CONFIG_PATH)

    # evaluate them
    evaluate_agent(teacher, 'wildlife', valid_data, MAIN_CONFIG_PATH)
    evaluate_agent(student, 'wildlife', valid_data2, MAIN_CONFIG_PATH)

if __name__ == '__main__':
    _main_()