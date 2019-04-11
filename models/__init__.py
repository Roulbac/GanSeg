import importlib
from models.base_model import BaseModel

def find_model(model_name):
    model_libname = 'models.{}_model'.format(model_name)
    modellib = importlib.import_module(model_libname)
    model = None
    modellib_classes = []
    for attr_str in dir(modellib):
        if(attr_str.startswith('_')):
            continue
        attr = getattr(modellib, attr_str)
        if issubclass(type(attr), type):  # Check if it is a class
            modellib_classes.append(attr)
    for cls in modellib_classes:
        if issubclass(cls, BaseModel):
            model = cls
    if model is None:
        print('No subclass of BaseModel found in {}'.format(model_libname))
        exit(0)
    return model


def get_model_parsing_modifier(model):
    model_class = find_model(model)
    return model_class.modify_commandline_options

def create_model(opt, visualizer=None):
    model = find_model(opt.model)
    model_instance = model(opt, visualizer=visualizer)
    print('Created instance of Model {}'.format(opt.model))
    if opt.load_epochiter != '':
        model_instance.restore_checkpoint(opt.load_epochiter)
    return model_instance
