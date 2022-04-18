import os
import time
import json

import torch
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from nets.retinanet import RetinaNet
from backbone import resnet50, LastLevelP6P7
from vis_utils.draw_box_utils import draw_box


def create_model(hparams):
    model = None
    if hparams.net == 'retinanet':
        backbone = resnet50(norm_layer=torch.nn.BatchNorm2d,
                            returned_layers=hparams.return_layers,
                            extra_blocks=LastLevelP6P7(256, 256),
                            trainable_layers=3,
                            hparams=hparams)
        model = RetinaNet(backbone, hparams.num_classes)

        weights_dict = torch.load(hparams.model_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)
    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(cfg):
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = create_model(cfg)
    model.to(device)

    # read class_indict
    label_json_path = cfg.cls
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}


    original_img = Image.open("./test.jpg")

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        draw_box(original_img,
                 predict_boxes,
                 predict_classes,
                 predict_scores,
                 category_index,
                 thresh=0.4,
                 line_thickness=3)
        plt.imshow(original_img)
        plt.show()
        # 保存预测的图片结果
        original_img.save("test_result.jpg")


if __name__ == '__main__':
    from params.infer_params import InferParams

    args = InferParams().create()
    main(args)

