import numpy as np
import pandas as pd


faculty_name = ['Arts', 'Social', 'Life', 'Medicine', 'Engineering', 'Humanities', 'Laws', 'Management', 'Exact']


def get_fac(fac):
    return faculty_name[int(fac) - 1]


def get_data():
    filename = 'ALL DATA EXPR 1_2_3 EXL.csv'

    df = pd.read_csv(filename)
    keys = df.keys()
    print(keys)
    relevant_keys = keys[[0,2,3,4,5,6,7,28,29,30,31,32,43]]
    stat_keys = range(-8,0)
    print(relevant_keys[stat_keys])
    df_relevant = df[relevant_keys]
    df_clean = df_relevant.copy() #df_relevant.dropna(subset=[relevant_keys[4]])
    return df_clean, relevant_keys, stat_keys


def get_faculty(df_clean, relevant_keys):
    # print(df_clean)
    faculty_list = np.unique(df_clean[relevant_keys[3]].dropna().values)
    n_faculty = faculty_list.shape[0]

    n_per_faculty = {}
    for fac1 in faculty_list:
        n_per_faculty[get_fac(fac1)] = df_clean[df_clean[relevant_keys[3]] == fac1].shape[0]
    return faculty_list, n_faculty


def get_results(df_clean, relevant_keys, stat_keys, faculty_list, n_faculty):
    results = {'faculty': {
        'mean': {},
        'z_stat': np.zeros(n_faculty),
        'p_val': np.zeros(n_faculty)
    }}
    for fac1 in faculty_list:
        results['faculty']['mean'][get_fac(fac1)] = {}
        for the_things in stat_keys:
            the_key = relevant_keys[the_things]
            fac1_data = df_clean[df_clean[relevant_keys[3]] == fac1][the_key]
            results['faculty']['mean'][get_fac(fac1)][the_key] = fac1_data.mean()