# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 12:05:14 2024

@author: dbda
"""
import pandas as pd
import os

os.chdir(r"C:\Users\dbda.STUDENTSDC\Desktop\project dummy")
fg = pd.read_csv('full_grouped.csv')

fg_cleaned = fg[(fg['Confirmed'] != 0) | (fg['Deaths'] != 0) | (fg['Recovered'] != 0) | (fg['Active'] != 0)
                | (fg['New cases'] != 0) | (fg['New recovered'] != 0) | (fg['New deaths'] != 0)]

fg_cleaned.to_csv('Full_Grouped_cleaned.csv', index = False)

FGC=fg_cleaned[(fg_cleaned['Confirmed'] >= 0) | (fg_cleaned['Deaths'] >= 0) | (fg_cleaned['Recovered'] >= 0) | (fg_cleaned['Active'] >= 0)
                | (fg_cleaned['New cases'] >= 0) | (fg_cleaned['New recovered'] >= 0) | (fg_cleaned['New deaths'] >= 0)]


FGC=FGC[(FGC['Confirmed'] > FGC["Active"]) & (FGC['Confirmed'] > FGC['Deaths'] ) & (FGC['Confirmed'] > FGC['Recovered'])]


FGC.to_csv('FGC.csv', index = False)
