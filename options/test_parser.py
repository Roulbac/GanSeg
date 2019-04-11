from options.options_parser import OptionsParser


class TestParser(OptionsParser):
    def __init__(self):
        super().__init__()
        parser = self.parser
        parser.add_argument(
            '--results_dir', type=str, default='./results', help='Directory where to store the results')
        parser.add_argument('--mode', type=str, default='test')
