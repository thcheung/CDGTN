from .CDGTN import CDGTN

def get_model(model_name, hidden_dim, classes, dropout, language):
    if model_name == 'CDGTN':
        return CDGTN(hidden_dim, classes,
                            dropout, language)