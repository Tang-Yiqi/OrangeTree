def LoadNewModel(configs):
    device = configs['device']
    ModelName = configs['model']['name']


    if ModelName == 'OrangeTree':
        from core import OrangeTree
        model = OrangeTree.OrangeTree(configs).to(device)
    elif ModelName == 'OrangeTreeFast':
        from core import OrangeTreeFast
        model = OrangeTreeFast.OrangeTree(configs).to(device)

    return model
