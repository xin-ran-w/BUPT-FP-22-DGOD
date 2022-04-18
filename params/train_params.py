import os

from .basic_params import BasicParams




class TrainParams(BasicParams):

    def __init__(self):
        super(TrainParams, self).__init__()
        self.add_params()

    def add_params(self):

        # the object detection framework to use
        self.parser.add_argument('--ni', type=int, default=0,
                                 help='the index of network, 0 for fasterrcnn')

        # source domain index
        self.parser.add_argument('--sdi', nargs='+', type=int, help='source domain index')

        # source domain index
        self.parser.add_argument('--tdi', type=int, help='target domain index')

        # num of class
        self.parser.add_argument('--num-classes', default=6, type=int, help='num of classes')

        # training algorithm
        self.parser.add_argument('--algorithm', default='Baseline', type=str,
                                 help='Train algorithm')

        # device
        self.parser.add_argument('--device', default='cuda', help='device')

        # batch size for each gpu
        self.parser.add_argument('-b', '--batch-size', default=4, type=int,
                                 help='images per gpu, the total batch size is $NGPU x batch_size')

        # max samples for each domain
        self.parser.add_argument('--max-samples', default=-1, type=int,
                                 help='-1 for using all samples, n > 0 for using n samples')

        # the iteration num of each domain
        self.parser.add_argument("--iter-num", default=-1,
                                 help="the iteration num of each domain, "
                                      "-1 use the iter num of the largest dataloader ")

        # start epoch
        self.parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')

        # total epoch
        self.parser.add_argument('--epochs', default=15, type=int, metavar='N',
                                 help='number of total epochs to run')

        # the probability to use augmentation
        self.parser.add_argument('--aug-prob', default=0.5, type=float,
                                 help='the probability to use augmentation')

        # learning rate
        self.parser.add_argument('--lr', default=0.001, type=float,
                                 help='initial learning rate, 0.02 is the default value for training '
                                      'on 8 gpus and 2 images_per_gpu')
        # SGD momentum
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                                 help='momentum')
        # SGD weight_decay
        self.parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                                 metavar='W', help='weight decay (default: 1e-4)',
                                 dest='weight_decay')

        # torch.optim.lr_scheduler.StepLR
        self.parser.add_argument('--lr-step-size', default=8, type=int,
                                 help='decrease lr every step-size epochs')

        # torch.optim.lr_scheduler.MultiStepLR
        self.parser.add_argument('--lr-steps', default=[7, 12], nargs='+', type=int,
                                 help='decrease lr every step-size epochs')

        # torch.optim.lr_scheduler.MultiStepLR
        self.parser.add_argument('--lr-gamma', default=0.1, type=float,
                                 help='decrease lr by a factor of lr-gamma')

        # print loss frequency
        self.parser.add_argument('--print-freq', default=50, type=int,
                                 help='print frequency')

        # save the model every n epochs
        self.parser.add_argument('--save-gap', type=int, default=5,
                                 help='save the model every n epochs')

        # weight save dir
        self.parser.add_argument('--output-dir', default='/data/wangxinran/model/ODDG/',
                                 help='path where to save')

        # train from last checkpoint
        self.parser.add_argument('--resume', default='',
                                 help='resume from checkpoint')

        # aspect-ratio-group-factor
        self.parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)

        # whether use mixed precision training
        self.parser.add_argument("--amp", default=False, help="use torch.cuda.amp for mixed precision training")

        # whether valid the train set
        self.parser.add_argument("--valid_train_set", default=False, help="valid the train set to judge over-fitting")



    def create(self):
        args = self.parser.parse_args()
        args.domain_info = self.domain_info
        args.source_domains = [os.path.basename(self.domain_info["data_dir"][i]) for i in args.sdi]
        args.target_domain = os.path.basename(self.domain_info["data_dir"][args.tdi])
        args.cls = args.domain_info['class_dict'][args.cfi]
        args.dist_url = "tcp://127.0.0.1:{}".format(args.port)
        args.net_info = self.net_info
        args.net = args.net_info['networks'][args.ni]
        args.pretraining_weight = args.net_info['weights'][args.ni]
        args.domain_num = len(args.source_domains)
        # self.print_args(args)
        return args
