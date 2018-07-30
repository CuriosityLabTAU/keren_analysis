#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

data_path = 'C:/Goren/CuriosityLab/Data/tablets/analysis/'

faculty_name = [unicode(x, 'utf-8') for x in ['תויונמא', 'הרבחה יעדמ', 'םייחה יעדמ', 'תואירבו האופר','הסדנה','חורה יעדמ', 'םיטפשמ', 'לוהינ', 'םיקיודמ םיעדמ']]
faculty_name_eng = ['Arts', 'Social', 'Life', 'Medicine', 'Engineering', 'Humanities', 'Laws', 'Management', 'Exact']


def get_fac(fac):
    return faculty_name[int(fac) - 1]


# load the data.
# from .csv to pandas data-frame
def load_data():
    filename = data_path + 'ALL_DATA_normalized_v001.csv'

    df = pd.read_csv(filename)
    return df


def file_name_to_subject(row):
    return row['file_name'].split('\\')[-1]

def rewarded_by_surprise(row):
    return row['adj_joy_cnt'] * row['adj_suprise_cnt']


# get only relevant information
def get_relevant_data(df):

    keys = df.keys()
    print('=== keys ===')
    for ik, kk in enumerate(keys):
        print(ik, kk)

    str_keys = ''
    for kk in keys:
        str_keys += "\'%s\', " % kk
    str_keys = str_keys[:-1]
    print(str_keys)
    all_the__keys_names = ['subject_number', 'file_name', 'experiment', 'condition_framing', 'condition_stop', 'pressed_stop', 'gender', 'faculty', 'faculty_1st', 'faculty_2nd', 'faculty_3rd', 'age', 'SAT', 'grades', 'curiosity_question_1', 'curiosity_question_2', 'curiosity_question_3', 'curiosity_question_4', 'curiosity_question_5', 'curiosity_question_6', 'curiosity_question_7', 'curiosity_question_8', 'curiosity_question_9', 'curiosity_question_10', 'curiosity_question_11', 'curiosity_question_12', 'curiosity_question_13', 'curiosity_question_14', 'curiosity_question_15', 'curiosity_question_16', 'curiosity_question_17', 'curiosity_question_18', 'curiosity_question_19', 'curiosity_question_20', 'curiosity_ques_stretching', 'curiosity_ques_embracing', 'curiosity_ques_embr_strt_TOTAL', 'BFI', 't0', 'total_listenning_time', 'normalized_total_listenning_time', 'Multi_discipline_entropy', 'transition_entropy', 'learning_1', 'learning_1__answer', 'learning_2', 'learning_2__answer', 'learning_3', 'learning_3__answer', 'learning_4', 'learning_4__answer', 'learning_5', 'learning_5__answer', 'learning_6', 'learning_6__answer', 'learning_7', 'learning_7__answer', 'learning_8', 'learning_8__answer', 'learning_9', 'learning_9__answer', 'learning_10', 'learning_10__answer', 'learning_questions_number', 'correct_learning_questions_number', 'correct_learning_questions_percent', 'wavs_amount', 'listening_per_faculty_med', 'listening_per_faculty_lif', 'listening_per_faculty_law', 'listening_per_faculty_art', 'listening_per_faculty_eng', 'listening_per_faculty_hum', 'listening_per_faculty_exa', 'listening_per_faculty_soc', 'listening_per_faculty_man']

    df['subject'] = df.apply(lambda row: file_name_to_subject(row), axis=1)

    relevant_keys = ['subject_number', 'subject', 'experiment', 'condition_framing', 'condition_stop', 'pressed_stop', 'gender', 'faculty', 'faculty_1st', 'faculty_2nd', 'faculty_3rd', 'age', 'SAT', 'grades',
                     'curiosity_ques_stretching', 'curiosity_ques_embracing', 'curiosity_ques_embr_strt_TOTAL', 'BFI', 't0', 'total_listenning_time', 'normalized_total_listenning_time', 'Multi_discipline_entropy',
                     'transition_entropy',
                     'learning_questions_number', 'correct_learning_questions_number', 'correct_learning_questions_percent', 'wavs_amount']


    # relevant_keys = keys[[0,2,3,4,5,6,7,28,29,30,31,42, 51, 54, 56, 58, 60, 62, 64, 66]]
    print('=== relevant keys ===')
    for ik, kk in enumerate(relevant_keys):
        print(ik, kk)
    df_relevant = df[relevant_keys]     # create a new dataframe, with only the relevant columns

    # create list of stat_keys
    # stat_keys = ['pressed_stop', 'SAT', 'grades',
    #                  'curiosity_ques_stretching', 'curiosity_ques_embracing', 'curiosity_ques_embr_strt_TOTAL', 'BFI', 't0', 'total_listenning_time', 'normalized_total_listenning_time', 'Multi_discipline_entropy',
    #                  'transition_entropy',
    #                  'learning_questions_number', 'correct_learning_questions_number', 'correct_learning_questions_percent', 'wavs_amount']
    stat_keys = [#'pressed_stop',
                 't0',
                 'wavs_amount', 'normalized_total_listenning_time',
                 'Multi_discipline_entropy', 'transition_entropy',
                 'correct_learning_questions_number', 'correct_learning_questions_percent',
                 'curiosity_ques_embr_strt_TOTAL', 'BFI']

    stat_names = ['T_0',
                  'N_{facts}', '\\overline{T}',
                  '\\overline{H}_m', '\\overline{H}_t',
                  'L', '\\overline{L}',
                  '\\overline{CEI}', '\\overline{OtE}']

    print('=== stat keys ===')
    for ik, kk in enumerate(stat_keys):
        print(ik, kk)

    # options to clean data
    df_clean = df_relevant.copy()
    # df_relevant.dropna(subset=[relevant_keys[4]])         # option 1: remove rows with no data
    # df_clean = df_clean[df_clean['experiment'] == 3]      # option 2: take only a single experiment

    # clean the data
    df_clean = df_clean[df_clean['curiosity_ques_embr_strt_TOTAL'] > 0.0]
    return df_clean, relevant_keys, stat_keys, stat_names


def renormalize_entropies(df_clean, stat_keys):
    for i_stat, x_str in enumerate(stat_keys):
        if 'entropy' in x_str:
            df_clean[x_str] = [np.min([0.99, v]) for v in df_clean[x_str].values]
            df_clean[x_str] = -np.log10(1.0 - df_clean[x_str].values)/2.0
    return df_clean


def add_affectiva(df_clean):
    df_aff = get_affectiva('C:/Goren/CuriosityLab/Research/Affectiva/summary_2502.xlsx')
    df_all = df_clean.join(df_aff.set_index('subject'), on='subject')
    df_all['feeling_curious'] = df_all.apply(lambda row: rewarded_by_surprise(row), axis=1)
    df_all['is_joy'] = df_all.apply(lambda row: float(row['Sum_of_Joy_Episodes'] > 0), axis=1)
    df_all['is_surprise'] = df_all.apply(lambda row: float(row['Sum_of_surprise_Episodes'] > 0), axis=1)
    df_all['is_smile'] = df_all.apply(lambda row: float(row['Sum_of_Smile_Episodes'] > 0), axis=1)
    print(len(df_all))
    df_all = df_all[df_all['Outlier'] == 0]
    print(len(df_all))
    print(df_all.columns)
    return df_all

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

def get_affectiva(filename):
    df_aff = pd.read_excel(filename)
    return df_aff