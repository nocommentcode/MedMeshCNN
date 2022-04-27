def create_model(opt):
    if opt.dataset_mode == 'regression':
        from .mesh_regression import RegressionModel
        model = RegressionModel(opt)
    else:
        from .mesh_classifier import ClassifierModel # todo - get rid of this ?
        model = ClassifierModel(opt)
    return model
