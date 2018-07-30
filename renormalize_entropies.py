from show_results import *
from pylab import *
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import statsmodels.api as sm


# first load the data
df = load_data()

# clean the data, get only relevant columns and information
df_clean, relevant_keys, stat_keys, stat_names = get_relevant_data(df)
print(df_clean.columns)
data = df_clean[['transition_entropy', 'Multi_discipline_entropy', 'wavs_amount']].dropna()
# print(data[data['transition_entropy'] == 1.0])
#
# u = np.sort(np.unique(data['transition_entropy']))
# print(u)
#
# data = data['transition_entropy']
# data[data > 0.99] = 0.99
# # data[data < 0.01] = 0.01
# renorm = -np.log10(1.0 - data.values) / 2.0

data['renorm'] = [np.min([0.99, v]) for v in data['Multi_discipline_entropy'].values]
data['renorm_log'] = -np.log10(1.0 - data['renorm'].values) / 2.0
data['renorm_inv'] = 1.0 / data['renorm'].values

fig = figure()
# ax = fig.add_subplot(121)
# ax.scatter(data['wavs_amount'], data['Multi_discipline_entropy'])
# ax = fig.add_subplot(122)
# ax.scatter(data['wavs_amount'], data['renorm'])
ax = fig.add_subplot(131)
bp = ax.hist(data['Multi_discipline_entropy'])
ax = fig.add_subplot(132)
bp = ax.hist(data['renorm_log'])
ax = fig.add_subplot(133)
bp = ax.hist(data['renorm_inv'])

# bp = ax.scatter(data.values, renorm, marker='x')

plt.show()
