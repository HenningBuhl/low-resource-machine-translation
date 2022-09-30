import torch


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3
