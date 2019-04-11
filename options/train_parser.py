from options.options_parser import OptionsParser


class TrainParser(OptionsParser):
    def __init__(self):
        super().__init__()
        parser = self.parser
        parser.add_argument('--no_viz', action='store_true',
                            help='Don\'t use visuals')
        parser.add_argument('--viz_is_remote', action='store_true',
                            help='Visdom server runs on a remote machine')
        parser.add_argument('--vizport', type=int, default='8096',
                            help='Visualiser\'s HTML server port')
        parser.add_argument('--vizaddr', type=str, default='localhost',
                            help='Visualiser\'s HTML server port')
        parser.add_argument('--init_type', type=str, default='xavier',
                            help='Weight init: normal, xavier, kaiming or orthogonal')
        parser.add_argument('--n_epochs', type=int,
                            default=200, help='Train for n epochs')
        parser.add_argument(
            '--lr', type=float, default=0.0002, help='Initial learning rate')
        parser.add_argument('--lr_policy', type=str, choices=[
            'plateau', 'step', 'lambda'], default='lambda', help='Learning rate schedueling policy')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='Decay learning rate by a gamma every lr_decay_iters')
        parser.add_argument('--decay_epoch', type=int, default=100,
                            help='Epoch after which start linear decay')
        parser.add_argument('--save_each', type=int, default=50,
                            help='Epoch saving frequency')
        parser.add_argument('--log_freq', type=int, default=200,
                            help='Epoch saving frequency')
        parser.add_argument('--print_freq', type=int, default=3,
                            help='Epoch saving frequency')
        parser.add_argument('--batch_size', type=int, default=1,
                            help='Batch size')
        parser.add_argument('--no_shuffle', action='store_true',
                            help='Shuffle data')
        parser.add_argument('--mode', type=str, default='train')
