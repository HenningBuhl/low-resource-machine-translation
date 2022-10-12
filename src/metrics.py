import torchmetrics


def get_bleu_metric():
    return torchmetrics.SacreBLEUScore(tokenize='13a', smooth=True)

def get_ter_metric():
    return torchmetrics.TranslationEditRate()

def get_chrf_metric():
    return torchmetrics.CHRFScore()
