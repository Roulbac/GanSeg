from options.test_parser import TestParser
from models import create_model, get_model_parsing_modifier
from datasets import create_dataset, get_dataset_parsing_modifier

parser = TestParser()
model_name = parser.get_model_name()
dataset_name = parser.get_dataset_name()

print('Model name: {}'.format(model_name))
print('Dataset name: {}'.format(dataset_name))

model_parser_modifier = get_model_parsing_modifier(model_name)
model_parser_modifier(parser, is_train=False)

dataset_parser_modifier = get_dataset_parsing_modifier(dataset_name)
dataset_parser_modifier(parser, is_train=False)

opts, _ = parser.parse_options()

opts_str = parser.make_opts_string(opts, verbose=True)

model = create_model(opts)

dataset = create_dataset(opts)

if opts.eval:
    model.set_eval()
model.test(dataset)
