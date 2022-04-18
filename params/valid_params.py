import os

from .basic_params import BasicParams

class ValidParams(BasicParams):
    
    def __init__(self):
        super(ValidParams, self).__init__()
        self._params()

    def _params(self):

        # the object detection framework to use
        self.parser.add_argument('--ni', type=int, default=0,
                                 help='the index of network, 0 for fasterrcnn')

        self.parser.add_argument('--sdi', nargs='+', type=int, help='source domain index')

        # source domain index
        self.parser.add_argument('--tdi', type=int, help='target domain index')

        # num of class
        self.parser.add_argument('--num-classes', default=8, type=int, help='num of classes')

        # training algorithm
        self.parser.add_argument('--algorithm', default='Baseline', type=str,
                                 help='algorithm')

        # batch size for validation
        self.parser.add_argument('-b', '--batch-size', default=4, type=int,
                                 help='batch size for validation')

        # max samples for each domain
        self.parser.add_argument('--max-samples', default=-1, type=int,
                                 help='-1 for using all samples, n > 0 for using n samples')

        # 使用设备类型
        self.parser.add_argument('--device', default='cuda:0', help='device')

        # model path
        self.parser.add_argument('--model-path', default=None, help='model path')

    def create(self):
        args = self.parser.parse_args()
        args.domain_info = self.domain_info
        args.target_domain = os.path.basename(self.domain_info["data_dir"][args.tdi])
        args.cls = args.domain_info['class_dict'][args.cfi]
        args.net_info = self.net_info
        args.net = args.net_info['networks'][args.ni]
        args.pretraining_weight = args.net_info['weights'][args.ni]
        args.domain_num = len(args.sdi)
        return args
