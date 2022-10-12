import torchmetrics


def get_bleu_metric():
    return 'bleu', torchmetrics.SacreBLEUScore(tokenize='13a', smooth=True)

def get_ter_metric():
    return 'ter', torchmetrics.TranslationEditRate()

def get_chrf_metric():
    return 'chrf', torchmetrics.CHRFScore()
