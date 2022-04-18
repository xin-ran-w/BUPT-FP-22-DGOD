import os

from .basic_params import BasicParams


class InferParams(BasicParams):

    def __init__(self):
        super(InferParams, self).__init__()
        self._params()

    def _params(self):

        # photos directory
        self.parser.add_argument('--dataset', type=str, default='bdd100k',
                                 help='the dataset name')

        # the object detection framework to use
        self.parser.add_argument('--ni', type=int, default=0,
                                 help='the index of network, 0 for fasterrcnn')

        # num of class
        self.parser.add_argument('--num-classes', default=8, type=int, help='num of classes')

        # training algorithm
        self.parser.add_argument('--algorithm', default='Baseline', type=str,
                                 help='algorithm')

        # batch size for validation
        self.parser.add_argument('-b', '--batch-size', default=4, type=int,
                                 help='batch size for validation')

        # device
        self.parser.add_argument('--device', default='cuda:0', help='device')

        # model path
        self.parser.add_argument('--model-path', default=None, help='model path')


    def create(self):
        args = self.parser.parse_args()
        args.domain_info = self.domain_info
        args.cls = args.domain_info['class_dict'][args.cfi]
        args.net = self.net_info['networks'][args.ni]
        return args




