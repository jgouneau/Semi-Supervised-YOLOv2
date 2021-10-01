import xml.etree.ElementTree as et
import json
import os
import cv2
from .core.utils import enable_memory_growth, list_images

def gen_xml(ann_path, xml_name, img_name, H, W, C, anns):
  # création de l'annotation
  # annotation
  annotation = et.Element("annotation")
  annotation.set("verified", "yes")
    # filename
  filename = et.SubElement(annotation, "filename")
  filename.text = img_name
    # size
  size = et.SubElement(annotation, "size")
      # width
  width = et.SubElement(size, "width")
  width.text = str(W)
      # height
  height = et.SubElement(size, "height")
  height.text = str(H)
      # depth
  depth = et.SubElement(size, "depth")
  depth.text = str(C)
      # pseudo-labelled
  pseudo_labelled = et.SubElement(annotation, "pseudo_labelled")
  pseudo_labelled.text = '1'
  for ann in anns:
    clas = ann[0]
    x_min = ann[1]
    y_min = ann[2]
    x_max = ann[3]
    y_max = ann[4]
    pl_conf = ann[5]
    # object
    obj = et.SubElement(annotation, "object")
      # name
    name = et.SubElement(obj, "name")
    name.text = clas
      # pl_confident
    pl_confident = et.SubElement(obj, "pl_confident")
    pl_confident.text = str(pl_conf)
      # bndbox
    bndbox = et.SubElement(obj, "bndbox")
        # xmin
    xmin = et.SubElement(bndbox, "xmin")
    xmin.text = str(x_min)
        # ymin
    ymin = et.SubElement(bndbox, "ymin")
    ymin.text = str(y_min)
        # xmax
    xmax = et.SubElement(bndbox, "xmax")
    xmax.text = str(x_max)
        # ymax
    ymax = et.SubElement(bndbox, "ymax")
    ymax.text = str(y_max)
  tree = et.ElementTree(annotation)
  tree.write(ann_path + xml_name)


def agent_gen_pseudo_labels(agent, dataset_name, threshold, main_config_path):
  enable_memory_growth()

  with open(main_config_path) as config_buffer:    
      main_config = json.loads(config_buffer.read())

  dataset_path = main_config['datasets_paths'][dataset_name]
  with open(dataset_path + "config.json") as config_buffer:    
      dataset_config = json.loads(config_buffer.read())
  
  agent_config_path = main_config['agents_config_path']
  with open(agent_config_path) as config_buffer:    
      agent_config = json.loads(config_buffer.read())
  
  labels = dataset_config['labels']

  base_path = "./YOLOv2/datasets/wildlife/trainset/unlabelled/img/" #dataset_config['trainset']['unlabelled']['img_folder']
  ann_path = "./YOLOv2/datasets/wildlife/trainset/unlabelled/ann/" #dataset_config['trainset']['unlabelled']['ann_folder']
  os.makedirs(ann_path, exist_ok=True)
  img_paths = list_images(base_path)
  for img_path in img_paths:
    img_name = img_path.split("/")[-1]
    file_name = img_name.split(".")[0]
    xml_name = file_name + ".xml"
    img = cv2.imread(img_path)
    H, W, C = img.shape
    boxes = agent.predict(img,
                            obj_threshold=agent_config['predict']['obj_threshold'],
                            nms_threshold=agent_config['predict']['nms_threshold'])
    anns = []
    for box in boxes:
      clas = labels[box.get_label()]
      xmin = int(box.xmin * W)
      ymin = int(box.ymin * H)
      xmax = int(box.xmax * W)
      ymax = int(box.ymax * H)
      pl_conf = 0
      if box.get_score() > threshold:
        pl_conf = 1
      ann = [clas, xmin, ymin, xmax, ymax, pl_conf]
      anns.append(ann)
    gen_xml(ann_path, xml_name, img_name, H, W, C, anns)