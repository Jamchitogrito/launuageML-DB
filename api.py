from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3


app = FastAPI()

with open('') as file:
    model = pickle.load(file)

with open('') as file:
    vectorizer = pickle.load(file)


def 