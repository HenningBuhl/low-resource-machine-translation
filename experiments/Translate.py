#!/usr/bin/env python
# coding: utf-8

# # Translate

# ## Setup

# ### Environment

# In[ ]:


# If this is a notebook which is executed in colab [in_colab=True]:
#  1. Mount google drive and use the repository in there [mount_drive=True] (the repository must be in your google drive root folder).
#  2. Clone repository to remote machine [mount_drive=False].
in_colab = False
mount_drive = True

try:
    # Check if running in colab.
    in_colab = 'google.colab' in str(get_ipython())
except:
    pass

if in_colab:
    if mount_drive:
        # Mount google drive and navigate to it.
        from google.colab import drive
        drive.mount('/content/drive')
        get_ipython().run_line_magic('cd', 'drive/MyDrive')
    else:
        # Pull repository.
        get_ipython().system('git clone https://github.com/HenningBuhl/low-resource-machine-translation')

    # Workaround for problem with undefined symbols (https://github.com/scverse/scvi-tools/issues/1464).
    get_ipython().system('pip install --quiet scvi-colab')
    from scvi_colab import install
    install()

    # Navigate to the repository and install requirements.
    get_ipython().run_line_magic('cd', 'low-resource-machine-translation')
    get_ipython().system('pip install -r requirements.txt')

    # Navigate to notebook location.
    get_ipython().run_line_magic('cd', 'experiments')


# In[ ]:


# Add src module directory to system path for subsecuent imports.
import sys
sys.path.insert(0, '../src')


# In[ ]:


from util import is_notebook

# Settings and module reloading (only in Jupyter Notebooks).
if is_notebook():
    # Module reloading.
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

    # Plot settings.
    get_ipython().run_line_magic('matplotlib', 'inline')


# ### Imports

# In[ ]:


# From packages.
import pytorch_lightning as pl
import argparse
from distutils.util import strtobool

# From repository.
from arg_manager import *
from constants import *
from data import *
from layers import *
from metric_logging import *
from plotting import *
from tokenizer import *
from transformer import *
from util import *


# ### Arguments

# In[ ]:


# Define arguments with argparse.
arg_manager = ArgManager()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Experiment.
parser.add_argument('--model-path', default='models/model', type=str, help='The path of the model to use.')
parser.add_argument('--text', default='example', type=str, help='The text to translate.')
parser.add_argument('--seed', default=0, type=int, help='The random seed of the program.')

# Parse args.
if is_notebook():
    sys.argv = ['-f']  # Used to make argparse work in jupyter notebooks (all args must be optional).
    args, _ = parser.parse_known_args()  # -f can lead to unknown argument.
else:
    args = parser.parse_args()


# ### Finalize

# In[ ]:


# Set seed.
from pytorch_lightning import seed_everything
seed_everything(args.seed, workers=True)


# ## Translate

# In[ ]:


# Load arguments.
m_args = load_dict(os.path.join(args.model_path, 'args.json'))


# In[ ]:


# Load tokenizers.
src_tokenizer = TokenizerBuilder(m_args.src_lang).build()
tgt_tokenizer = TokenizerBuilder(m_args.tgt_lang).build()


# In[ ]:


# Load model.
model = load_model_from_path(args.model_path, src_tokenizer, tgt_tokenizer)
model = model.to(device)


# In[ ]:


# Translate.
#args.text = 'adjust me'
translation = model.translate(args.text, method='greedy')
print(f'Translated:\n\t{args.text}\nTo:\n\t{translation}')


# In[ ]:




