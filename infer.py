import os
import time
import json

import fiftyone as fo
import torch
from PIL import Image

from torchvision.transforms import functional as TF
from tqdm import tqdm

from algorithms import get_algorithm_class
from vis_utils.load_samples import convert_to_relative_value, convert_to_xywh
from val_utils.create_model import create_model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def predict(dataset, cfg):
    device = torch.device(cfg.device)
    weight = cfg.model_path
    assert os.path.exists(weight), "{} file dose not exist.".format(weight)
    print("Using device: {}.".format(device))
    print("Using model: {}.".format(weight))

    # create model
    model = create_model(cfg)
    model.load_state_dict(torch.load(weight, map_location=device)["model"])
    model.to(device)
    model.eval()

    # read class_dict
    label_json_path = cfg.cls
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}

    inference_time_list = []

    with torch.no_grad():
        for index, sample in enumerate(tqdm(dataset)):

            image = Image.open(sample.filepath)
            image = TF.to_tensor(image).to(device)
            _, height, width = image.shape
            image = torch.unsqueeze(image, dim=0)

            t_start = time_synchronized()
            predictions = model(image.to(device))[0]

            t_end = time_synchronized()
            inference_time = t_end - t_start
            inference_time_list.append(inference_time)

            boxes = predictions["boxes"].to("cpu").numpy()
            labels = predictions["labels"].to("cpu").numpy()
            scores = predictions["scores"].to("cpu").numpy()

            # print(boxes)
            # exit()

            detections = []
            for label, score, box in zip(labels, scores, boxes):
                box = convert_to_xywh(box.tolist())
                rel_box = convert_to_relative_value(box,
                                                    img_width=width,
                                                    img_height=height)
                # rel_box = box.tolist()
                detections.append(
                    fo.Detection(
                        label=category_index[label],
                        bounding_box=rel_box,
                        confidence=score
                    )
                )
            sample[cfg.algorithm] = fo.Detections(detections=detections)
            sample.save()

    mean_inference_time = sum(inference_time_list) / len(inference_time_list)
    print("Total {}, Mean inference time: {}".format(len(inference_time_list), mean_inference_time))


def main(params):
    algorithm_class = get_algorithm_class(params.algorithm)
    params = algorithm_class.get_hparams(params)
    # params.domain_num = 4
    name = '{}_{}c'.format(params.dataset, params.num_classes)
    dataset = fo.load_dataset(name)
    predict(dataset, params)
    print("Finished adding predictions, you can reload the dataset to see it.")


if __name__ == '__main__':
    # results = dataset.evaluate_detections(
    #     pred_field="faster_rcnn",
    #     gt_field="ground_truth",
    #     eval_key="eval",
    #     compute_mAP=True,
    #     method='coco'
    # )
    from params.infer_params import InferParams

    args = InferParams().create()
    main(args)
