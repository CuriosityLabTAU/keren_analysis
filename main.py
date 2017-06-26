from analyze_data import *
from show_results import *

df = load_data()
df_clean, relevant_keys, stat_keys = get_relevant_data(df)
faculty_list, n_faculty = get_faculty(df_clean, relevant_keys)
df_clean = get_campus(df_clean, faculty_list)
# results = get_results(df_clean, relevant_keys, stat_keys, faculty_list, n_faculty)

# faculty_groups = find_faculty_clusters(df_clean, relevant_keys, stat_keys, faculty_list)
# df_clean = add_faculty_group(df_clean, faculty_list, faculty_groups, 'cluster')
# analyze_faculty(df_clean, relevant_keys, stat_keys) #, faculty_columns=['faculty_group_cluster'])
# show_faculty(df_clean, relevant_keys, stat_keys, faculty_list)
analyze_multi_regression_cat(df_clean, relevant_keys)
# analyze_regression(df_clean, relevant_keys, stat_keys)
# analyze_factor_analysis(df, relevant_keys, stat_keys)

# analyze_faculty_preferences(df)

# normality_testing(df_clean, relevant_keys, stat_keys)

# bucketing(df_clean, relevant_keys, stat_keys)

# characterizing(df_clean, relevant_keys, stat_keys)