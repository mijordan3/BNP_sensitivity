import numpy as np

def get_one_hot(targets, nb_classes):
    # TODO: test this
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])
