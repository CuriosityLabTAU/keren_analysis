#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

data_path = '../keren_data/'

faculty_name = [unicode(x, 'utf-8') for x in ['תויונמא', 'הרבחה יעדמ', 'םייחה יעדמ', 'תואירבו האופר','הסדנה','חורה יעדמ', 'םיטפשמ', 'לוהינ', 'םיקיודמ םיעדמ']]
faculty_name_eng = ['Arts', 'Social', 'Life', 'Medicine', 'Engineering', 'Humanities', 'Laws', 'Management', 'Exact']


def get_fac(fac):
    return faculty_name[int(fac) - 1]


# load the data.
# from .csv to pandas data-frame
def load_data():
    filename = data_path + 'ALL_DATA_10_normalized.csv'

    df = pd.read_csv(filename)
    return df


# get only relevant information
def get_relevant_data(df):

    keys = df.keys()
    print('=== keys ===')
    for ik, kk in enumerate(keys):
        print(ik, kk)

    relevant_keys = keys[[0,2,3,4,5,6,7,28,29,30,31,42, 51, 54, 56, 58, 60, 62, 64, 66]]
    print('=== relevant keys ===')
    for ik, kk in enumerate(relevant_keys):
        print(ik, kk)
    df_relevant = df[relevant_keys]     # create a new dataframe, with only the relevant columns

    # create list of stat_keys
    stat_keys = [5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # stat_keys = [13, 14, 15, 16, 17, 18, 19]
    print('=== stat keys ===')
    print(relevant_keys[stat_keys])

    # options to clean data
    df_clean = df_relevant.copy()
    # df_relevant.dropna(subset=[relevant_keys[4]])         # option 1: remove rows with no data
    # df_clean = df_clean[df_clean['experiment'] == 3]      # option 2: take only a single experiment
    return df_clean, relevant_keys, stat_keys

def get_faculty(df_clean, relevant_keys):
    # 3 = faculty
    # .values --> convert from dataframe to numpy array
    faculty_list = np.unique(df_clean[relevant_keys[3]].dropna().values)

    # -- remove some faculties, if you want --
    # faculty_list = np.delete(faculty_list, 7)   # remove management

    n_faculty = faculty_list.shape[0]

    # how many rows per faculty
    n_per_faculty = {}
    for fac1 in faculty_list:
        n_per_faculty[get_fac(fac1)] = df_clean[df_clean[relevant_keys[3]] == fac1].shape[0]

    return faculty_list, n_faculty

def get_campus(df_clean, faculty_list):
    faculty_groups = {
        'cei':
            {
                'group1': [3, 4],
                'group2': [2, 7],
                'group3': [0, 1, 5, 6]
            },
        'campus':
            {
                'east': [2, 3, 4, 7],
                'west': [0, 1, 5, 6]
            },
        'multi':
            {
                'group1': [0, 4, 6],
                'group2': [7, 5, 2],
                'group3': [1, 3]
            },
        't0':
            {
                'group1': [6],
                'group2': [2, 5, 1, 3, 4],
                'group3': [0, 7]
            },
        'transition':
            {
                'group1': [1, 0, 3],
                'group2': [7, 2, 4,6],
                'group3': [5]
            }
    }

    for g_name, groups in faculty_groups.items():
        for i, l in groups.items():
            for j in l:
                df_clean.loc[df_clean['faculty'] == faculty_list[j], 'faculty_group_' + g_name] = i
    return df_clean


def add_faculty_group(df_clean, faculty_list, faculty_group, group_name):
    for i, l in faculty_group.items():
        for j in l:
            df_clean.loc[df_clean['faculty'] == faculty_list[j], 'faculty_group_' + group_name] = i
    return df_clean


def get_results(df_clean, relevant_keys, stat_keys, faculty_list, n_faculty):
    results = {'faculty': {
        'mean': {},
        'z_stat': np.zeros(n_faculty),
        'p_val': np.zeros(n_faculty)
    }}
    # go over each faculty
    for fac1 in faculty_list:
        results['faculty']['mean'][get_fac(fac1)] = {}
        # go over all stats that we're intereseted in
        for the_things in stat_keys:
            the_key = relevant_keys[the_things]
            fac1_data = df_clean[df_clean[relevant_keys[3]] == fac1][the_key]
            results['faculty']['mean'][get_fac(fac1)][the_key] = fac1_data.mean()
            # TODO Keren: put here all the calculation, and do the plotting elsewhere
            # work in progress

    return results