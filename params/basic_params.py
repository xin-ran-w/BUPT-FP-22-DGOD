import argparse


def print_args(args):
    for key, value in sorted(vars(args).items()):
        if isinstance(value, dict):
            print('{}'.format(key).center(40, '-'))
            for sub_key, sub_value in value.items():
                print(sub_key, '=', sub_value)
        else:
            print(key, '=', value)
    print("experiment config".center(80, '='))

class BasicParams(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser(description=__doc__)

        # use index to control which domain you want to use.

        self.domain_info = {
            "data_dir": [
                "/data/wangxinran/dataset/object_detection/bdd100k",            # domain 0
                "/data/wangxinran/dataset/object_detection/cityscapes",         # domain 1
                "/data/wangxinran/dataset/object_detection/foggycity",          # domain 2
                "/data/wangxinran/dataset/object_detection/kitti",              # domain 3
                "/data/wangxinran/dataset/object_detection/sim10k",             # domain 4
            ],

            "class_dict": [
                './class_dict/bcf.json',                                        # class_dict 0
                './class_dict/bcfks.json',                                      # class_dict 1
                './class_dict/bcf_b.json',                                      # class_dict 2 with background
                './class_dict/bcfks_b.json',                                    # class_dict 3 with background
            ],
        }

        self.net_info = {
            "networks": [
                'fasterrcnn',
                'retinanet',
            ],

            "weights": [
                '/data/wangxinran/weight/fasterrcnn_resnet50_fpn_coco.pth',
                '/data/wangxinran/weight/retinanet_resnet50_fpn_coco.pth',
            ]
        }

        # device
        self.parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                                 help='number of data loading workers (default: 4)')

        # class_dict
        self.parser.add_argument('--cfi', default=0, type=int,
                                 help='choose the class dict default is bcf.json')

        # process num
        self.parser.add_argument('--world-size', default=4, type=int,
                                 help='number of distributed processes')

        # the url for distributed learning
        self.parser.add_argument('--port', type=int, default=28000, help='url used to set up distributed training')

    def create(self):
        raise NotImplementedError
