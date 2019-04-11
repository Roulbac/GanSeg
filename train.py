from utils.visualization import Visualizer
from options.train_parser import TrainParser
from models import create_model, get_model_parsing_modifier
from utils.visualization import update_progressbar
from datasets import create_dataset, get_dataset_parsing_modifier, get_loader

parser = TrainParser()
model_name = parser.get_model_name()
dataset_name = parser.get_dataset_name()
print('Model name: {}'.format(model_name))
print('Dataset name: {}'.format(dataset_name))

model_parser_modifier = get_model_parsing_modifier(model_name)
model_parser_modifier(parser, is_train=True)

dataset_parser_modifier = get_dataset_parsing_modifier(dataset_name)
dataset_parser_modifier(parser, is_train=True)

opts, _ = parser.parse_options()


opts_str = parser.make_opts_string(opts, verbose=True)

if opts.no_viz:
    viz = None
else:
    viz = Visualizer(port=opts.vizport,
                     hostname=opts.vizaddr,
                     is_remote=opts.viz_is_remote)


model = create_model(opts, viz)
t_dataset, v_dataset = create_dataset(opts)
t_loader = get_loader(data=t_dataset,
                      batch_size=opts.batch_size,
                      shuffle=not opts.no_shuffle,
                      num_workers=opts.num_workers
                      )

model.init_viz(opts_str)

for n in range(model.epoch + 1, opts.n_epochs + 1):
    print('Epoch {}'.format(n))
    iters_p_epoch = len(t_loader)
    curr_iter = 0
    for example in t_loader:
        model.set_input(example)
        model.optimize_parameters()
        model.iter += 1
        curr_iter += 1
        if curr_iter % opts.print_freq == 0 \
           or curr_iter == iters_p_epoch:
            update_progressbar(curr_iter,
                               iters_p_epoch, model.traincache)
        if model.iter % opts.log_freq == 0:
            model.save_traincache()
            model.do_validation(v_dataset)
            model.update_viz()
    model.epoch = n
    model.update_learning_rate()
    if model.epoch % opts.save_each == 0:
        model.save_checkpoint()
