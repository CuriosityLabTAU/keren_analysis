import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from our_stats import *
from analyze_data import *
from show_results import *

df_clean, relevant_keys, stat_keys = get_data()
faculty_list, n_faculty = get_faculty(df_clean, relevant_keys)
results = get_results(df_clean, relevant_keys, stat_keys, faculty_list, n_faculty)

# analyze_faculty(df_clean, relevant_keys, stat_keys, faculty_list, results)
show_faculty(df_clean, relevant_keys, stat_keys, faculty_list)
# analyze_multi_regression(df_clean, relevant_keys)
# analyze_regression(df_clean, relevant_keys, stat_keys)
