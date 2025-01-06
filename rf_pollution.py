import tensorflow_decision_forests as tfdf
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math

!gdown 13zMPxc9Wwg5jxj-W162kpqUjpQ8cdlRR

df = pd.read_csv("/content/pollution.csv")
df.head()
