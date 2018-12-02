from analyze_data import *
from show_results import *

# first load the data
df = load_data()

# clean the data, get only relevant columns and information
df_clean, relevant_keys, stat_keys, stat_names = get_relevant_data(df)

# df_clean = renormalize_entropies(df_clean, stat_keys)

# re-save the data
# df_clean.to_csv(data_path + 'df_clean_unnormalized_entropies.csv')

# df_clean = add_affectiva(df_clean)

# get some lists of faculty
# faculty_list, n_faculty = get_faculty(df_clean, relevant_keys)

# divide faculties according to campus / and other
# !!! don't use
# df_clean = get_campus(df_clean, faculty_list)

# === get some results =====

# !!! faculty related
# -- basic statistics, currently per faculty
# results = get_results(df_clean, relevant_keys, stat_keys, faculty_list, n_faculty)
# -- use k-means clustering algorithm to find faculty groups
# faculty_groups = find_faculty_clusters(df_clean, relevant_keys, stat_keys, faculty_list)
# -- add the new groups to the dataframe
# df_clean = add_faculty_group(df_clean, faculty_list, faculty_groups, 'cluster')

# IMPORTANT: how to do multicomparison analysis
# analyze_faculty(df_clean, relevant_keys, stat_keys, faculty_columns=['faculty_group_cluster'])

# IMPORTANT: how to draw bar plots
# show_faculty(df_clean, relevant_keys, stat_keys, faculty_list)

# -- factor analysis
# analyze_factor_analysis(df, relevant_keys, stat_keys)

# analyze_faculty_preferences(df)

# !!! multi-linear regression
# analyze_multi_regression_cat(df_clean, relevant_keys)

# -- performs the shapiro normality test for each stat
# also draw histograms
# normality_testing(df_clean, relevant_keys, stat_keys)


# -- tries to find clusters of people according to specific parameters
# characterizing(df_clean, relevant_keys, stat_keys)

# === less important ===

# analyze_regression(df_clean, relevant_keys, stat_keys)

# bucketing(df_clean, relevant_keys, stat_keys)


# NEW: conditions
# analyze_conditions(df_clean, stat_keys, stat_names)

# analyze_correlations(df_clean, stat_keys)

# go_wild(df_clean)

paper_assessment(df_clean)

# clusters(df_clean)

# basic_stats(df_clean)

# affectiva_groups(df_clean)
