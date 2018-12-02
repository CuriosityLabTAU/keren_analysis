import rpy2
print(rpy2.__version__)
# import rpy2's package module
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
import pandas as pd
import rpy2.robjects.lib.ggplot2 as ggplot2
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import numpy as np

# import R's utility package
utils = rpackages.importr('utils')
# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the list
rprint = robjects.globalenv.get("print")

# the LPA package and functions
tidyLPA = rpackages.importr('tidyLPA')
estimate_profiles = robjects.r.estimate_profiles
plot_profiles = robjects.r.plot_profiles
compare_solutions = robjects.r.compare_solutions
attributes = robjects.r.attributes

# get the data
pandas2ri.activate()
data_path = 'C:/Goren/CuriosityLab/Data/tablets/analysis/'
# df = pd.read_csv(data_path + 'df_clean_unnormalized_entropies.csv')
df = pd.read_csv(data_path + 'df_clean.csv')
print(df.columns)

# set if want partial data: only non-stop
# df = df[df['condition_stop'] == 0]
# df = df[df['condition_framing'] == 1]

# first find models and profiles
c = compare_solutions(df,
                      # 'wavs_amount',
                      'normalized_total_listenning_time',
                      # 'correct_learning_questions_percent',
                      'Multi_discipline_entropy',
                      # 'transition_entropy'
                      )
n_profile = 3
the_model = 1
# plot profiles
y = estimate_profiles(df,
                      # 'wavs_amount',
                      'normalized_total_listenning_time',
                      # 'correct_learning_questions_percent',
                      'Multi_discipline_entropy',
                      # 'transition_entropy',
                      n_profiles = n_profile, model = the_model,
                      print_which_stats='all',
                      # center_raw_data=True, scale_raw_data=True
                      )
p = plot_profiles(y)
# rprint(y)

y = estimate_profiles(df,
                      # 'wavs_amount',
                      'normalized_total_listenning_time',
                      # 'correct_learning_questions_percent',
                      'Multi_discipline_entropy',
                      # 'transition_entropy',
                      n_profiles=n_profile, model=the_model,
                      print_which_stats='all',
                      # center_raw_data=True, scale_raw_data=True,
                      return_orig_df=True)
y_df = pandas2ri.ri2py(y)
# get parameters
# m = estimate_profiles(df,
#                       'normalized_total_listenning_time', 'Multi_discipline_entropy',
#                       'transition_entropy', 'wavs_amount', n_profiles=5, model=2,
#                       print_which_stats='all', to_return='mclust',
#                       center_raw_data=True, scale_raw_data=True, return_orig_df=True)
# rprint(attributes(m))

print('==========================')
the_measure = 'SAT'

data= pd.merge_ordered(y_df[['subject_number', 'profile']], df, on=['subject_number'])
print(data.columns)
print(len(df), len(data))

final_data = data[[the_measure, 'profile']].dropna()

# formula = ' %s ~ C(profile)' % the_measure
#
# est = smf.ols(formula=formula, data=final_data).fit()
# print(est.summary())
# print('F(%d, %d) = %2.2f, p=%2.3f' % (est.df_model, est.df_resid, est.fvalue, est.f_pvalue))
#
# table = sm.stats.anova_lm(est, typ=2)  # Type 2 ANOVA DataFrame
# print(table)

groups = [final_data[final_data['profile']==str(p+1)][the_measure] for p in range(n_profile)]
print('Group:')
for g in groups:
    print('size: ', len(g), the_measure, ':', np.median(g))
print(stats.kruskal(groups[0], groups[1], groups[2]))

posthoc = sp.posthoc_dunn(final_data, val_col=the_measure, group_col='profile')
print(posthoc)
#
# profile = [float(p) for p in final_data['profile'].values]
# print(pairwise_tukeyhsd(final_data[the_measure], profile))

print('The End!')