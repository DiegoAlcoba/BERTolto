# suppress warnings
import warnings;
warnings.filterwarnings('ignore');

# common imports
import pandas as pd
from pathlib import Path
import sqlite3
import os
import subprocess
import sys
from transformers import AutoTokenizer, logging

import numpy as np
import math
import re
import glob
import os
import sys
import json
import random
import pprint as pp
import textwrap

import logging

import spacy
import nltk
import textacy

from tqdm.auto import tqdm
# register `pandas.progress_apply` and `pandas.Series.map_apply` with `tqdm`
tqdm.pandas()

# pandas display options
# https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html#available-options
pd.options.display.max_columns = 30 # default 20
pd.options.display.max_rows = 60 # default 60
pd.options.display.float_format = '{:.2f}'.format
# pd.options.display.precision = 2
pd.options.display.max_colwidth = 200 # default 50; -1 = all
# otherwise text between $ signs will be interpreted as formula and printed in italic
pd.set_option('display.html.use_mathjax', False)