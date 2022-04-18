import os
from xml.etree import ElementTree

import fiftyone as fo


def read_xml(txt_path):
    with open(txt_path) as fid:
        xml_str = fid.read()
    xml = ElementTree.fromstring(xml_str)
    return xml

def read_txt(txt_path, anno_dir):
    with open(txt_path) as read:
        anno_list = [os.path.join(anno_dir, line.strip())
                     for line in read.readlines() if len(line.strip()) > 0]
    return anno_list


def convert_to_xywh(bbox):

    bbox[2] -= bbox[0]
    bbox[3] -= bbox[1]

    return bbox


def convert_to_relative_value(bbox, img_width, img_height):

    bbox[0] /= img_width
    bbox[1] /= img_height
    bbox[2] /= img_width
    bbox[3] /= img_height

    return bbox


def parse_xml_to_dict(xml):

    if len(xml) == 0:
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def create_fo_dataset(samples, name="my-detection-dataset"):
    dataset = fo.Dataset(name)
    dataset.add_samples(samples)
    return dataset

def load_samples(img_dir, anno_dir, samples_names_file):
    samples = []
    annos = read_txt(samples_names_file, anno_dir)

    for anno in annos:
        anno = read_xml(os.path.join(anno_dir, anno))
        anno = parse_xml_to_dict(anno)["annotation"]
        if '.jpg' in anno['filename']:
            img_path = os.path.join(img_dir, anno['filename'])
        else:
            img_path = os.path.join(img_dir, str(anno['filename']) + '.jpg')
        sample = fo.Sample(filepath=img_path)
        size = anno['size']
        img_width = float(size['width'])
        img_height = float(size['height'])

        detections = []
        for obj in anno['object']:
            category = obj["name"]

            bbox_dict = obj['bndbox']
            bbox = [
                float(bbox_dict['xmin']),
                float(bbox_dict['ymin']),
                float(bbox_dict['xmax']),
                float(bbox_dict['ymax'])
            ]

            bbox = convert_to_xywh(bbox=bbox)
            bbox = convert_to_relative_value(bbox=bbox,
                                             img_width=img_width,
                                             img_height=img_height)
            
            detections.append(
                fo.Detection(label=category, bounding_box=bbox)
            )
        sample["ground_truth"] = fo.Detections(detections=detections)
        samples.append(sample)
    return samples
