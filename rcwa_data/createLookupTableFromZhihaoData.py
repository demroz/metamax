#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:03:05 2024

@author: maxzhelyeznyakov
"""

import mat73
import pandas as pd
data = mat73.loadmat('result.mat')['result']

r1 = data[:,0]
wave = data[:,1]
phase = data[:,2]
amp = data[:,3]

datad = {}
datad['r1'] = r1
datad['wave'] = wave
datad['theta'] = wave*0
datad['phi'] = wave*0
datad['t'] = amp
datad['p'] = phase

df = pd.DataFrame(datad)
df.to_csv('zhihao-lookup-table.csv')

