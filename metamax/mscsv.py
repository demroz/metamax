#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:41:53 2024

@author: maxzhelyeznyakov
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
csv = pd.read_csv('../data/hyperboloid/ms.csv')
#%%
r = csv['r'].to_numpy().reshape([5000,5000])
plt.imshow(r)

np.savetxt('../data/hyperboloid/hyperboloid.csv', r)