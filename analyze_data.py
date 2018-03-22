#!/usr/bin/python
# -*- coding: utf-8 -*-
from get_data import *
from scipy import stats
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.decomposition import factor_analysis
import matplotlib.pyplot as plt
import seaborn as sns
from csv import *
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from stamp.plugins.multiGroups.postHoc.Scheffe import Scheffe
from sklearn.cluster import KMeans
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols


# df_clean, relevant_keys, stat_keys = get_data()
# faculty_list, n_faculty = get_faculty(df_clean, relevant_keys)
# results = get_results(df_clean, relevant_keys, stat_keys, faculty_list, n_faculty)


def find_faculty_clusters(df_clean, relevant_keys, stat_keys, faculty_list):
    x = np.zeros([len(faculty_list), len(stat_keys)])
    for ifac, fac in enumerate(faculty_list):
        for istat, the_things in enumerate(stat_keys):
            the_key = relevant_keys[the_things]
            x[ifac, istat] = np.mean(df_clean[df_clean['faculty'] == fac][the_key].dropna().values)

    num_clusters = 4
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x)
    print(faculty_name_eng)
    print(kmeans.labels_)
    faculty_groups = {}
    for i in range(0, num_clusters):
        x = list([j for j in range(0, len(kmeans.labels_)) if kmeans.labels_[j] == i])
        faculty_groups['group' + str(i)] = [j-1 for j in x]
    return faculty_groups


def analyze_faculty(df_clean, relevant_keys, stat_keys, faculty_columns=None):
    if faculty_columns is None:
        faculty_columns = [u'faculty_group_multi'] #,
       # u'faculty_group_t0', u'faculty_group_campus', u'faculty_group_cei',
       # u'faculty_group_transition']
       #  faculty_columns = [u'faculty']

    for faculty_column in faculty_columns:
        faculty_list = np.unique(df_clean[faculty_column].dropna().values)
        print('================== ', faculty_column, ' ==================')
        for the_things in stat_keys:
            the_key = relevant_keys[the_things]
            print('--------- ', the_key, '----------')

            # ==== one-side anova
            fac_data = []       # a list of numpy array of all the data from the faculty (per key)
            for fac1 in faculty_list:
                fac_data.append(df_clean[df_clean[faculty_column] == fac1][the_key].dropna().values)
            if len(fac_data) == 2:
                print(stats.f_oneway(fac_data[0], fac_data[1]))
            elif len(fac_data) == 3:
                print(stats.f_oneway(fac_data[0], fac_data[1], fac_data[2]))
            elif len(fac_data) == 4:
                print(stats.f_oneway(fac_data[0], fac_data[1], fac_data[2], fac_data[3]))
            else:
                print(stats.f_oneway(fac_data[0], fac_data[1], fac_data[2],
                               fac_data[3], fac_data[4], fac_data[5],
                               fac_data[6], fac_data[7], fac_data[8]))
            # # ==== scheffe
            # print('scheffe')
            # preferences = {}
            # preferences['Pseudocount'] = 0.5
            # preferences['Executable directory'] = '.'
            # preferences['Replicates'] = 1000
            # scheffe = Scheffe(preferences)
            # pValues, effectSize, lowerCI, upperCI, labels, _ = scheffe.run(fac_data, 0.95, [str(x) for x in faculty_list])
            # for i, l in enumerate(labels):
            #     print(l, pValues[i])


            # ==== tukey
            fac_data = []
            fac_group = []
            for fac1 in faculty_list:
                x = df_clean[df_clean[faculty_column] == fac1][the_key].dropna().values
                fac_data.extend(x)
                fac_group.extend([fac1 for i in range(0, len(x))])

            mc = MultiComparison(fac_data, fac_group)
            result = mc.tukeyhsd()
            print(result.summary())
            print(mc.groupsunique)
            result.plot_simultaneous()
            plt.show()


        #     n_fac1_data = fac1_data.shape[0]
        #     for fac2 in faculty_list:
        #         fac2_data = df_clean[df_clean[faculty_column] == fac2][the_key].dropna()
        #         n_fac2_data = fac2_data.shape[0]
        #         z_stat, p_val = stats.ranksums(fac1_data, fac2_data)
        #
        #         if p_val < 0.05:
        #             print(the_key, n_fac1_data, n_fac2_data,
        #                   get_fac(fac1), get_fac(fac2), z_stat, p_val,
        #                   results['faculty']['mean'][get_fac(fac1)][the_key],
        #                   results['faculty']['mean'][get_fac(fac2)][the_key])


def analyze_multi_regression(df_clean, relevant_keys):
    free_exp_stats = [relevant_keys[k] for k in [1, 11, 19]]
    label_stat = [relevant_keys[k] for k in [9]]
    regression_stats = [relevant_keys[k] for k in [1, 11, 19,9]]
    print(regression_stats)
    df_reg = df_clean[regression_stats].dropna()

    lr = linear_model.LinearRegression()
    x = df_reg[free_exp_stats].values
    y = df_reg[label_stat].values

    # cross_val_predict returns an array of the same size as `y` where each entry
    # is a prediction obtained by cross validation:
    lr.fit(x, y)
    print(lr.coef_)
    predicted = lr.predict(x)
    # predicted = cross_val_predict(lr, x, y, cv=10)

    x2 = sm.add_constant(x)
    est = sm.OLS(y, x2)
    est2 = est.fit()
    print(est2.summary())


    # fig, ax = plt.subplots()
    # ax.scatter(y, predicted)
    # ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    # ax.set_xlabel('Measured')
    # ax.set_ylabel('Predicted')
    # plt.show()

def analyze_multi_regression_cat(df_clean, relevant_keys):
    free_exp_stats = [relevant_keys[k] for k in [1, 6, 10, 11]]
    label_stat = [relevant_keys[k] for k in [7, 8, 9]]

    # get only the relevant columns for this specific analysis
    # doing this because of .dropna, want to eliminate only rows with but RELEVANT inforamtion
    regression_stats = [relevant_keys[k] for k in [1, 6, 10, 11, 7, 8, 9]]
    print('regression_stats', regression_stats)
    df_reg = df_clean[regression_stats].dropna()    # new "clean" dataframe .dropna

    # the name of the columns include spaces ' ', '+'
    # this is BAD, so replace spaces ' ' with underscore '_'
    new_columns = []
    for c in df_reg.columns:
        new_columns.append(c.replace(' ', '_').replace('+', '_'))
    df_reg.columns = new_columns
    print(df_reg.columns)

    # print the dataframe and only a sample of the data for each column
    print('head', df_reg.head())
    # !!! multi-linear regression
    # formula (what_you_want_to_predict ~ parameters_that_can_explain_them
    # if the parameter is CATEGORICAL (e.g experiment), just put it in C()
    est = smf.ols(formula="curiosity_ques_stretching ~ transition_entropy + Multi_discipline_entropy + C(experiment)", data=df_reg).fit()
    print(est.summary())
    # lr.fit(x, y)
    # print(lr.coef_)
    # predicted = lr.predict(x)
    # # predicted = cross_val_predict(lr, x, y, cv=10)
    #
    # x2 = sm.add_constant(x)
    # est = sm.OLS(y, x2)
    # est2 = est.fit()
    # print(est2.summary())


def analyze_regression(df_clean, relevant_keys, stat_keys):
    reg_result = ","
    for the_things1 in stat_keys:
        reg_result += relevant_keys[the_things1] + ","
    reg_result += "\n"
    for the_things1 in stat_keys:
        reg_result += relevant_keys[the_things1] + ","
        the_key1 = relevant_keys[the_things1]
        for the_things2 in stat_keys:
            the_key2 = relevant_keys[the_things2]
            if the_key1 == the_key2:
                keys_df = df_clean[[the_key1]].dropna()
                data_key1 = keys_df[:][the_key1]
                z_stat, p_val = stats.ranksums(data_key1, data_key1)
                print(the_key1, the_key2, data_key1.shape, data_key1.shape, z_stat, p_val)
                reg_result += "(" + str(1.0) + ";" + str(0.0) + "), "
            else:
                keys_df = df_clean[[the_key1, the_key2]].dropna()
                # keys_df.to_csv(the_key1 + the_key2 + '.csv')
                print(keys_df.head())
                data_key1 = keys_df[the_key1]
                data_key2 = keys_df[the_key2]
                slope, intercept, r_value, p_value, std_err = stats.linregress(data_key1.values, data_key2.values)
                print(the_key1, the_key2, data_key1.shape, data_key2.shape, r_value, p_value)
                if p_value < 0.05 and the_things1 > the_things2:
                    sns.regplot(x=data_key1, y=data_key2)
                    plt.show()
                reg_result += "(" + '{0:.3f}'.format(r_value) + ";" + '{0:.3f}'.format(p_value) + "), "
        reg_result += "\n"
    f = open('regression_results.csv', 'w')
    f.write(reg_result)
    f.close()


def analyze_factor_analysis(df, relevant_keys, stat_keys):
    # take all the things you want to put in factors
    fa_keys = range(8,18)   # curiosity question
    # fa_keys.append(31)
    # fa_keys.append(42)
    # fa_keys.append(66)
    print('fa_keys', df.keys()[fa_keys])
    df_fa = df[fa_keys].dropna()

    # go over several number of possible factors
    # we don't know a-priori how many factors there should be
    n_components = range(1, len(fa_keys))
    fa_scores = []
    for n in n_components:
        fa = factor_analysis.FactorAnalysis()   # only define the object, not running yet
        fa.n_components = n                     # how many factors
        fa.fit(df_fa.values)                    # only here runs the fit
        fa_scores.append(fa.score(df_fa.values))
        if n == 2:
            print(fa.components_)   # the components, should show us how to collect into factors
            # TODO keren: check with Michal's explanation on how to do that
    print(n_components, fa_scores)
    plt.plot(n_components, fa_scores)
    plt.xlabel('number of components')
    plt.ylabel('score')
    plt.show()
    # print(fa)
    # # print(fa.get_precision())
    # print(fa.get_covariance().shape)
    # print(fa.score(df_fa.values))


def analyze_faculty_preferences(df):
    faculty_list = np.unique(df['faculty'].dropna().values)
    eng_to_heb = {
        'med': 3,
        'lif': 2,
        'law': 6,
        'art': 0,
        'eng': 4,
        'hum': 5,
        'exa': 8,
        'soc': 1,
        'man': 7
    }

    prop_keys = []
    for k in df.keys():
        if 'listening per faculty:' in k:
            prop_keys.append(k)
            print(k, faculty_name[eng_to_heb[k.split(':')[1]]])

    pref_results = ','
    for fac in prop_keys:
        pref_results += fac.split(':')[1] + ','
    pref_results += 'interested in myself compared to others'
    pref_results += '\n'
    results = np.zeros([len(faculty_list), len(prop_keys)])
    for ifac, fac in enumerate(prop_keys):
        f = faculty_list[eng_to_heb[fac.split(':')[1]]]
        x = df[df['faculty'] == f][prop_keys].values
        y = np.nanmean(x, axis=0)
        y[np.isnan(y)] = 0
        pref_results += fac.split(':')[1] + ','
        for i in range(y.shape[0]):
            if np.isnan(y[i]):
                pref_results += '0,'
            else:
                pref_results += str(int(y[i])) + ','
        this_one = y[ifac]
        others = np.sum(y) - this_one
        mean_others = others / (y.shape[0] - 1)
        pref_results += str(this_one / mean_others)
        pref_results += '\n'
        results[ifac,:] = y


    print(pref_results)

    f = open('results/faculty_preference_results.csv', 'w')
    f.write(pref_results.encode('utf-8'))
    f.close()

# ----- General analysis ----

# summary --> they are ALL normally distributed
def normality_testing(df_clean, relevant_keys, stat_keys):
    for s in stat_keys:
        x = df_clean[relevant_keys[s]].dropna().values
        # this is the shapiro normality test
        w, p = stats.shapiro(x)
        # TODO keren: check what does it mean to pass the normality test!
        print(x.shape, relevant_keys[s], w, p, p<w)
        plt.hist(x, 50, facecolor='green', alpha=0.75)
        plt.title(relevant_keys[s])
        plt.show()

def bucketing(df_clean, relevant_keys, stat_keys):
    for the_things1 in stat_keys:
        the_key1 = relevant_keys[the_things1]
        for the_things2 in stat_keys:
            the_key2 = relevant_keys[the_things2]
            if the_key1 != the_key2 and (the_key1.find('mean') == -1 or the_key2.find('mean') == -1):
                keys_df = df_clean[[the_key1, the_key2]].dropna()
                # keys_df.to_csv(the_key1 + the_key2 + '.csv')
                # print(keys_df.head())
                data_key1 = keys_df[the_key1].values
                data_key2 = keys_df[the_key2].values
                n_groups = 4
                perc = [np.percentile(data_key1, 100.0 * float(x) / float(n_groups)) for x in range(n_groups+1)]

                # get the proper groups, divide according to key one, values according to key 2
                groups = []
                for kp in range(len(perc)-1):
                    ind1 = np.where(data_key1 >= perc[kp])[0]
                    ind2 = np.where(data_key1 < perc[kp+1])[0]
                    ind = list(set(ind1).intersection(ind2))
                    groups.append(data_key2[ind])

                # f_value, p_value = stats.f_oneway(groups[0], groups[1], groups[2])
                # if p_value < 0.05:
                #     print(the_key1, the_key2, groups[0].shape, groups[1].shape, groups[1].shape, f_value, p_value)

                    # ==== tukey
                fac_data = []
                fac_group = []
                for ig, gr in enumerate(groups):
                    g = gr.tolist()
                    fac_data.extend(g)
                    fac_group.extend([ig for i in range(0, len(g))])

                mc = MultiComparison(fac_data, fac_group)
                result = mc.tukeyhsd()
                if True in result.reject:
                    print(the_key1, the_key2)
                    print(result.summary())
                    print(mc.groupsunique)
                    if n_groups > 2:
                        result.plot_simultaneous()
                        plt.title(the_key1 + " -- " + the_key2)
                        plt.show()
                    # f_value, p_value = stats.f_oneway(groups[0], groups[1])
                    # print(the_key1, the_key2, groups[0].shape, groups[1].shape, f_value, p_value)



def characterizing(df_clean, relevant_keys, stat_keys, num_clusters=5):
    curiosity_stats = ['normalized total listenning time', 'curiosity_ques_embr+strt TOTAL', 'Multi discipline entropy', 'BFI'] #, 'engagement mean', 'valence mean', 'surprise mean']
    x = df_clean[curiosity_stats].dropna().values
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x)
    print(kmeans.cluster_centers_)




import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy import stats


def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq'] / sum(aov['sum_sq'])
    return aov


def omega_squared(aov):
    mse = aov['sum_sq'][-1] / aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq'] - (aov[:-1]['df'] * mse)) / (sum(aov['sum_sq']) + mse)
    return aov



from matplotlib.pyplot import *

labels = ["Baseline", "System"]
data =   [3.75               , 4.75]
error =  [0.3497             , 0.3108]

def condition_plot(a, b, x, a_str, b_str, x_str, df_, x_name):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 76}
    matplotlib.rc('font', **font)

    x_stop_0 = x[b == 0].dropna()
    x_stop_1 = x[b == 1].dropna()
    x_priming_0 = x[a == 0].dropna()
    x_priming_1 = x[a == 1].dropna()

    print('Stop: No: %2.2f \\pm  %2.2f, N=%d, Yes: %2.2f \\pm %2.2f, N=%d' %
          (np.mean(x_stop_0), np.std(x_stop_0), len(x_stop_0),
           np.mean(x_stop_1), np.std(x_stop_1), len(x_stop_1)))
    print('Priming: No: %2.2f \\pm  %2.2f, N=%d, Yes: %2.2f \\pm %2.2f, N=%d' %
          (np.mean(x_priming_0), np.std(x_priming_0), len(x_priming_0),
           np.mean(x_priming_1), np.std(x_priming_1), len(x_priming_1)))
    print()

    stop_stat, stop_p = stats.ttest_ind(x_stop_0, x_stop_1)
    priming_stat, priming_p = stats.ttest_ind(x_priming_0, x_priming_1)

    if x_name == 'T':
        ab_lbl = ['NO-STOP', 'STOP']
        x_data = []
        x_data.append(x_stop_0)
        x_data.append(x_stop_1)
        n_x = 2
        x_cond = np.array([1, 2])
    else:
        ab_lbl = ['NO-STOP/NO-PRIMING','NO-STOP/PRIMING', 'STOP/NO-PRIMING',  'STOP/PRIMING']
        x_data = []
        x_data.append(x[(a==0) & (b==0)])
        x_data.append(x[(a==1) & (b==0)])
        x_data.append(x[(a == 0) & (b == 1)])
        x_data.append(x[(a == 1) & (b == 1)])
        n_x = 4
        x_cond = np.array([1, 2, 3, 4])

    x_avg = np.zeros([n_x])
    x_std = np.zeros([n_x])
    for i_d, x_d in enumerate(x_data):
        x_avg[i_d] = np.mean(x_d)
        x_std[i_d] = np.std(x_d) / np.sqrt(len(x_d))

    data = x_avg
    xlocations = x_cond
    error = x_std
    labels = ab_lbl

    # df_['ab'] = df_.apply(lambda row: row[a_str] + row[b_str] * 2, axis=1)
    # df_na = df_[[x_str, 'ab']].dropna()
    # mc = MultiComparison(df_na[x_str], df_na['ab'])
    # result = mc.tukeyhsd()
    # print(result.summary())
    # print(mc.groupsunique)

    xlocations = np.array(range(len(data))) + 0.5
    width = 0.5
    the_linewidth = 10
    bar(xlocations, data, yerr=error, width=width, ecolor='black')


    # if stop_p < 0.05:
    #     plot([xlocations[0] + 0.25, xlocations[1] + 0.25], [max(data)+max(error) * 1.2, max(data)+max(error) * 1.2],
    #          '-', color='black', linewidth=the_linewidth)
    #     plot([xlocations[2] + 0.25, xlocations[3] + 0.25], [max(data) + max(error) * 1.2, max(data) + max(error) * 1.2],
    #          '-', color='black', linewidth=the_linewidth)
    #     plot([(xlocations[0] + 0.25 + xlocations[1] + 0.25) / 2.0,
    #           (xlocations[0] + 0.25 + xlocations[1] + 0.25) / 2.0,],
    #          [max(data) + max(error) * 1.2, max(data) + max(error) * 1.7],
    #          '-', color='black', linewidth=the_linewidth)
    #     plot([(xlocations[2] + 0.25 + xlocations[3] + 0.25) / 2.0,
    #           (xlocations[2] + 0.25 + xlocations[3] + 0.25) / 2.0, ],
    #          [max(data) + max(error) * 1.2, max(data) + max(error) * 1.7],
    #          '-', color='black', linewidth=the_linewidth)
    #     plot([(xlocations[0] + 0.25 + xlocations[1] + 0.25) / 2.0,
    #           (xlocations[2] + 0.25 + xlocations[3] + 0.25) / 2.0, ],
    #          [max(data) + max(error) * 1.7, max(data) + max(error) * 1.7],
    #          '-', color='black', linewidth=the_linewidth)
    #     if stop_p < 0.001:
    #         text(np.mean(xlocations) + 0.25, max(data) + max(error) * 2.0, '***')
    #     elif stop_p < 0.01:
    #         text(np.mean(xlocations) + 0.25, max(data) + max(error) * 2.0, '**')
    #     elif stop_p < 0.05:
    #         text(np.mean(xlocations) + 0.25, max(data) + max(error) * 2.0, '*')
    #
    # if priming_p < 0.05:
    #     plot([xlocations[0] + 0.35, xlocations[2] + 0.15], [max(data) + max(error) * 1.2, max(data) + max(error) * 1.2],
    #          '-', color='black', linewidth=the_linewidth)
    #     plot([xlocations[1] + 0.35, xlocations[3] + 0.15], [max(data) + max(error) * 1.5, max(data) + max(error) * 1.5],
    #          '-', color='black', linewidth=the_linewidth)
    #     plot([(xlocations[0] + 0.25 + xlocations[2] + 0.25) / 2.0,
    #           (xlocations[0] + 0.25 + xlocations[2] + 0.25) / 2.0, ],
    #          [max(data) + max(error) * 1.2, max(data) + max(error) * 2.0],
    #          '-', color='black', linewidth=the_linewidth)
    #     plot([(xlocations[1] + 0.25 + xlocations[3] + 0.25) / 2.0,
    #           (xlocations[1] + 0.25 + xlocations[3] + 0.25) / 2.0, ],
    #          [max(data) + max(error) * 1.5, max(data) + max(error) * 2.0],
    #          '-', color='black', linewidth=the_linewidth)
    #     plot([(xlocations[0] + 0.25 + xlocations[2] + 0.25) / 2.0,
    #           (xlocations[1] + 0.25 + xlocations[3] + 0.25) / 2.0, ],
    #          [max(data) + max(error) * 2.0, max(data) + max(error) * 2.0],
    #          '-', color='black', linewidth=the_linewidth)
    #     if priming_p < 0.001:
    #         text(np.mean(xlocations) + 0.25, max(data) + max(error) * 2.3, '***')
    #     elif priming_p < 0.01:
    #         text(np.mean(xlocations) + 0.25, max(data) + max(error) * 2.3, '**')
    #     elif priming_p < 0.05:
    #         text(np.mean(xlocations) + 0.25, max(data) + max(error) * 2.3, '*')

    # yticks(range(0, 8))
    # xticks(xlocations + width / 2, labels)
    plt.xticks([], [])

    xlim(0, xlocations[-1] + width * 2)
    title_str = '%s, p_stop=%2.6f, p_priming=%2.6f' % (x_str, stop_p, priming_p)
    # title(title_str)
    # gca().get_xaxis().tick_bottom()
    # gca().get_yaxis().tick_left()
    # axis([xlocations[0], xlocations[-1] + 0.5, 0, max(data) + max(error) * 4.0])
    # y_label = '$%s$' % x_name
    # ylabel(y_label, usetex=True, fontsize=48)
    plt.tick_params(axis='both', which='major', labelsize=48)
    show()

    # print(x_avg, x_std)
    # plt.bar(x_cond, x_avg)
    # plt.errorbar(x_cond, x_avg, yerr=x_std, fmt='.', ecolor='black')
    # plt.xticks(x_cond, ab_lbl)
    # plt.title(x_str)
    # plt.axis([0,5, 0, 1.0])
    # plt.show()





def analyze_conditions(df_clean, stat_keys, stat_names):
    a_str = 'condition_framing'
    b_str = 'condition_stop'
    c_str = 'gender'

    for i_stat, x_str in enumerate(stat_keys):
        x_name = stat_names[i_stat]
        print('=============', x_str, '==================')

        x = df_clean[x_str].dropna()
        print('%s: %2.2f \\pm %2.2f, N=%d' % (x_str, np.mean(x), np.std(x), len(x)))
        t_stat, p_value = stats.ttest_1samp(x, 0)
        print('T-score: %2.3f, p_value: %2.3f' % (t_stat, p_value))

        data = df_clean[[a_str, b_str, x_str]].dropna()
        a = data[a_str]
        b = data[b_str]
        x = data[x_str]

        formula = '%s ~ C(%s) * C(%s)' % (x_str, a_str, b_str)
        est = smf.ols(formula=formula, data=data).fit()
        print(est.summary())
        print('F(%d, %d) = %2.2f, p=%2.3f' % (est.df_model, est.df_resid, est.fvalue, est.f_pvalue))

        print('====== anova ======')
        moore_lm = ols(formula=formula, data=data).fit()
        table = sm.stats.anova_lm(moore_lm, typ=2)  # Type 2 ANOVA DataFrame
        print(table)
        print('====== end ======')

        # # formula = '%s ~ C(%s) * C(%s) * C(%s)' % (x_str, a_str, b_str, c_str)
        # model = ols(formula, data).fit()
        # aov_table = anova_lm(model, typ=2)
        # eta_squared(aov_table)
        # omega_squared(aov_table)
        # print(aov_table)
        # fig = interaction_plot(a, b, x, colors=['red', 'blue'], markers=['D', '^'], ms=10)
        # plt.title(x_str)
        # plt.show()

        formula = '%s ~ C(%s)' % (x_str, a_str)
        est = smf.ols(formula=formula, data=data).fit()
        print(est.summary())

        # # formula = '%s ~ C(%s) * C(%s) * C(%s)' % (x_str, a_str, b_str, c_str)
        # model = ols(formula, data).fit()
        # aov_table = anova_lm(model, typ=2)
        # eta_squared(aov_table)
        # omega_squared(aov_table)
        # print(aov_table)

        formula = '%s ~ C(%s)' % (x_str, b_str)
        est = smf.ols(formula=formula, data=data).fit()
        print(est.summary())


        # model = ols(formula, data).fit()
        # aov_table = anova_lm(model, typ=2)
        # eta_squared(aov_table)
        # omega_squared(aov_table)
        # print(aov_table)


        # the_data = x
        # the_group = a + 2 * b
        # f_value, p_value = stats.f_oneway(the_data, the_group)
        # print(f_value, p_value)
        # mc = MultiComparison(the_data, the_group)
        # result = mc.tukeyhsd()
        # print(result.summary())
        # print(mc.groupsunique)
        # result.plot_simultaneous()
        # plt.show()

        condition_plot(a, b, x, a_str, b_str, x_str, data, x_name)


        # except:
        #     plt.show()
    # get 2x2 conditions
    # https://www.marsja.se/three-ways-to-carry-out-2-way-anova-with-python/




def analyze_correlations(df_clean, stat_keys):
    y_str = 'curiosity_ques_embr_strt_TOTAL'
    for x_str in stat_keys:
        data = df_clean[[x_str, y_str, 'condition_framing', 'condition_stop']].dropna()

        # x = data[data[x_str] > 0][x_str]
        # y = data[data[x_str] > 0][y_str]

        x = data[x_str]
        y = data[y_str]

        est = smf.ols(formula="%s ~ %s + C(condition_framing) + C(condition_stop)" % (y_str, x_str), #
                      data=data).fit()
        try:

            print(est.summary())
            slope, intercept, r_value, p_value, std_err = stats.linregress(x.values, y.values)
            sns.regplot(x=x, y=y)
            plt.title('r=%2.3f, p=%2.3f' % (r_value, p_value))
            plt.xlabel(x_str)
            plt.show()
        except:
            pass


def go_wild(df_clean):
    # est = smf.ols(formula="BI ~ normalized_total_listenning_time * Multi_discipline_entropy * transition_entropy", +
    df_current = df_clean#[df_clean['condition_stop'] == 1]

    est = smf.ols(formula="curiosity_ques_embr_strt_TOTAL ~ wavs_amount * Multi_discipline_entropy + C(condition_framing)", # + wavs_amount + Multi_discipline_entropy",
                  data=df_current).fit()
    print(est.summary())


def clusters(df_clean):
    num_clusters = 3
    curiosity_stats = ['correct_learning_questions_percent', 'Multi_discipline_entropy', 'transition_entropy'] #, 'transition_entropy'] #, 'adj_suprise_cnt'] #'normalized_total_listenning_time',
    # curiosity_stats = ['adj_joy_cnt', 'adj_suprise_cnt']
    x = df_clean[curiosity_stats].dropna()
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x[curiosity_stats].values)
    print('=======================================================')
    print(curiosity_stats)
    print(kmeans.cluster_centers_)

    the_measures = ['curiosity_ques_embr_strt_TOTAL', 'SAT', 'BFI', 'grades']

    for the_measure in the_measures:
        print()
        print('-----', the_measure, '------')
        y = df_clean[[the_measure] + curiosity_stats].dropna()

        # for c in curiosity_stats:
        #     est = smf.ols(
        #         formula="SAT ~ correct_learning_questions_percent + condition_framing + condition_stop + transition_entropy + Multi_discipline_entropy",
        #         data=df_clean).fit()
        # print(c, est.summary())
        # print()

        y_label = kmeans.predict(y[curiosity_stats].values)
        df_clean['labels'] = -1
        df_clean.loc[y.index.values, 'labels'] = y_label

        df_labeled = df_clean[df_clean['labels'] >= 0]

        formula = '%s ~ C(labels)' % the_measure

        # model = ols(formula, df_labeled).fit()
        # aov_table = anova_lm(model, typ=2)
        # eta_squared(aov_table)
        # omega_squared(aov_table)
        # print(aov_table)

        print('====== anova ======')
        moore_lm = ols(formula=formula, data=df_labeled).fit()
        table = sm.stats.anova_lm(moore_lm, typ=2)  # Type 2 ANOVA DataFrame
        print(table)
        print('====== end ======')

        draw_data_avg = np.zeros([3])
        draw_data_std = np.zeros([3])
        map_cluster_to_group = [0, 1, 2]
        for n in range(num_clusters):
            the_group = df_labeled[df_labeled['labels'] == n][the_measure].values
            print('group %d: %2.2f \\pm %2.2f, N=%d' % (n, np.mean(the_group), np.std(the_group), len(the_group)))
            draw_data_avg[map_cluster_to_group[n]] = np.mean(the_group)
            draw_data_std[map_cluster_to_group[n]] = np.std(the_group) / np.sqrt(len(the_group))

        for n in range(num_clusters):
            the_group = df_labeled[df_labeled['labels'] == n][the_measure].values
            for m in range(num_clusters):
                if m > n:
                    the_other_group = df_labeled[df_labeled['labels'] == m][the_measure].values
                    t_stat, p_value = stats.ttest_ind(the_group, the_other_group)
                    print(n, m, t_stat, p_value)

        if num_clusters > 2:
            mc = MultiComparison(df_labeled[the_measure], df_labeled['labels'])
            result = mc.tukeyhsd()
            print(result.summary())
            print(mc.groupsunique)
            result.plot_simultaneous()
            plt.xlabel(the_measure)
            plt.ylabel('groups')
            plt.title('N = %d' % len(df_labeled))
            plt.show()

            font = {'family': 'normal',
                    'weight': 'bold',
                    'size': 76}
            matplotlib.rc('font', **font)

            xlocations = np.array(range(len(draw_data_avg))) + 0.5
            data = draw_data_avg
            error = draw_data_std
            width = 0.5
            the_linewidth = 10
            labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']
            bar(xlocations, data, yerr=error, width=width, ecolor='black')
            # if the_measure == 'SAT':
            #     plot([xlocations[0] + 0.25, xlocations[0] + 0.25],
            #          [max(data) + max(error) * 1.2, max(data) + max(error) * 2.0],
            #          '-', color='black', linewidth=the_linewidth)
            #     plot([xlocations[1] + 0.25, xlocations[1] + 0.25],
            #          [max(data) + max(error) * 1.2, max(data) + max(error) * 2.0],
            #          '-', color='black', linewidth=the_linewidth)
            #     plot([xlocations[0] + 0.25, xlocations[1] + 0.25],
            #          [max(data) + max(error) * 2.0, max(data) + max(error) * 2.0],
            #          '-', color='black', linewidth=the_linewidth)
            #     text(np.mean(xlocations[[0,1]]) + 0.25, max(data) + max(error) * 2.3, '***')
            #
            #     plot([xlocations[0] + 0.25, xlocations[0] + 0.25],
            #          [max(data) + max(error) * 1.2, max(data) + max(error) * 2.5],
            #          '-', color='black', linewidth=the_linewidth)
            #     plot([xlocations[2] + 0.25, xlocations[2] + 0.25],
            #          [max(data) + max(error) * 1.2, max(data) + max(error) * 2.5],
            #          '-', color='black', linewidth=the_linewidth)
            #     plot([xlocations[0] + 0.25, xlocations[2] + 0.25],
            #          [max(data) + max(error) * 2.5, max(data) + max(error) * 2.5],
            #          '-', color='black', linewidth=the_linewidth)
            #     text(np.mean(xlocations[[0, 2]]) + 0.25, max(data) + max(error) * 2.7, '***')
            #     ylabel('SAT', usetex=True, fontsize=48)
            # else:
            #     plot([xlocations[0] + 0.25, xlocations[0] + 0.25],
            #          [max(data) + max(error) * 1.2, max(data) + max(error) * 2.0],
            #          '-', color='black', linewidth=10)
            #     plot([xlocations[1] + 0.25, xlocations[1] + 0.25],
            #          [max(data) + max(error) * 1.2, max(data) + max(error) * 2.0],
            #          '-', color='black', linewidth=the_linewidth)
            #     plot([xlocations[0] + 0.25, xlocations[1] + 0.25],
            #          [max(data) + max(error) * 2.0, max(data) + max(error) * 2.0],
            #          '-', color='black', linewidth=the_linewidth)
            #     text(np.mean(xlocations[[0, 1]]) + 0.25, max(data) + max(error) * 2.3, '***')
            #
            #     plot([xlocations[1] + 0.25, xlocations[1] + 0.25],
            #          [max(data) + max(error) * 1.2, max(data) + max(error) * 2.5],
            #          '-', color='black', linewidth=the_linewidth)
            #     plot([xlocations[2] + 0.25, xlocations[2] + 0.25],
            #          [max(data) + max(error) * 1.2, max(data) + max(error) * 2.5],
            #          '-', color='black', linewidth=the_linewidth)
            #     plot([xlocations[1] + 0.25, xlocations[2] + 0.25],
            #          [max(data) + max(error) * 2.5, max(data) + max(error) * 2.5],
            #          '-', color='black', linewidth=the_linewidth)
            #     text(np.mean(xlocations[[1, 2]]) + 0.25, max(data) + max(error) * 2.7, '***')
            #     ylabel('$\overline{CEI}$', usetex=True, fontsize=48)

            # xticks(xlocations + width / 2, labels)
            xticks([], [])
            plt.tick_params(axis='both', which='major', labelsize=48)
            xlim(0, xlocations[-1] + width * 2)

            axis([np.min(xlocations[0]), np.max(xlocations)+0.5, 0, max(data) + max(error) * 10.0])
            show()



    y = df_clean[['curiosity_ques_embr_strt_TOTAL','SAT'] + curiosity_stats].dropna()
    y_label = kmeans.predict(y[curiosity_stats].values)
    df_clean['labels'] = -1
    df_clean.loc[y.index.values, 'labels'] = y_label
    df_labeled = df_clean[df_clean['labels'] >= 0]

    group_markers = [u'+', u'x', u'o']
    markers = [group_markers[i] for i in df_labeled['labels'].values]
    x1 = df_labeled['curiosity_ques_embr_strt_TOTAL'].values
    x2 = df_labeled['SAT'].values

    for _s, _x, _y in zip(markers, x1, x2):
        plt.scatter(_x, _y, marker=_s)
    plt.show()


def basic_stats(df_clean):
    print('Participants: %d' % len(df_clean))

    age = df_clean['age'].dropna().values
    print('Age: %d, average %2.3f, std %2.3f' % (len(age), np.mean(age), np.std(age)))

    gender = df_clean['gender'].dropna().values
    print('Gender: %d , males %d, females %d' % (len(gender), np.sum(gender == 1), np.sum(gender == 2)))


def assessment(df_clean):
    data = df_clean[['curiosity_ques_embr_strt_TOTAL', 'BFI']].dropna()
    est = smf.ols(formula="curiosity_ques_embr_strt_TOTAL ~  BFI", data=data).fit()
    print(est.summary())

    data = df_clean[df_clean['SAT'] > 0][['curiosity_ques_embr_strt_TOTAL', 'SAT']].dropna()
    est = smf.ols(formula="curiosity_ques_embr_strt_TOTAL ~  SAT", data=data).fit()
    print(est.summary())

    data = df_clean[df_clean['grades'] > 0][['curiosity_ques_embr_strt_TOTAL', 'grades']].dropna()
    est = smf.ols(formula="curiosity_ques_embr_strt_TOTAL ~  grades", data=data).fit()
    print(est.summary())

    data = df_clean[['curiosity_ques_embr_strt_TOTAL', 'correct_learning_questions_percent', 'condition_framing', 'condition_stop']].dropna()
    est = smf.ols(formula="curiosity_ques_embr_strt_TOTAL ~  correct_learning_questions_percent + condition_framing + condition_stop", data=data).fit()
    print(est.summary())

    data = df_clean[['SAT', 'correct_learning_questions_percent', 'condition_framing', 'condition_stop']].dropna()
    est = smf.ols(formula="SAT ~  correct_learning_questions_percent + condition_framing + condition_stop", data=data).fit()
    print(est.summary())

    data = df_clean[['curiosity_ques_embr_strt_TOTAL', 'correct_learning_questions_percent', 'Multi_discipline_entropy', 'transition_entropy', 'condition_stop', 'condition_framing']].dropna()
    est = smf.ols(formula="curiosity_ques_embr_strt_TOTAL ~  correct_learning_questions_percent + transition_entropy + Multi_discipline_entropy + condition_framing + condition_stop", data=data).fit()
    print(est.summary())


    data = df_clean[['SAT', 'correct_learning_questions_percent', 'Multi_discipline_entropy',
                     'transition_entropy', 'condition_stop', 'condition_framing']].dropna()
    est = smf.ols(
        formula="SAT ~  correct_learning_questions_percent + Multi_discipline_entropy + transition_entropy + condition_framing + condition_stop",
        data=data).fit()
    print(est.summary())


    data = df_clean[['Multi_discipline_entropy',
                     'transition_entropy']].dropna()
    est = smf.ols(
        formula="transition_entropy ~  Multi_discipline_entropy",
        data=data).fit()
    print(est.summary())

    data = df_clean[['SAT', 'wavs_amount', 'Multi_discipline_entropy', 'correct_learning_questions_percent']].dropna()
    data = data[data['SAT'] > 500]
    est = smf.ols(
        formula="SAT ~  correct_learning_questions_percent + wavs_amount + Multi_discipline_entropy",
        data=data).fit()
    print(est.summary())


    data = df_clean[['curiosity_ques_embr_strt_TOTAL', 'wavs_amount', 'Multi_discipline_entropy', 'correct_learning_questions_percent']].dropna()
    est = smf.ols(
        formula="curiosity_ques_embr_strt_TOTAL ~  correct_learning_questions_percent + wavs_amount + Multi_discipline_entropy",
        data=data).fit()
    print(est.summary())


    data = df_clean[['BFI', 'wavs_amount', 'Multi_discipline_entropy',
                     'correct_learning_questions_percent']].dropna()
    est = smf.ols(
        formula="BFI ~  correct_learning_questions_percent + wavs_amount + Multi_discipline_entropy",
        data=data).fit()
    print(est.summary())

    data = df_clean[['grades', 'wavs_amount', 'Multi_discipline_entropy',
                     'correct_learning_questions_percent']].dropna()


    est = smf.ols(
        formula="grades ~  correct_learning_questions_percent + wavs_amount + Multi_discipline_entropy",
        data=data).fit()
    print(est.summary())


def affectiva_groups(df_clean):
    # parameter = 'Multi_discipline_entropy'
    # variable = 'is_joy'
    #
    # df_current = df_clean[df_clean['over_50'] == 1]
    #
    # df_current = df_current[[parameter, variable]].dropna()
    #
    # the_group = df_current[df_clean[variable] == 0][parameter].values
    # the_other_group = df_current[df_clean[variable] == 1][parameter].values
    # n = len(the_group)
    # m = len(the_other_group)
    #
    # t_stat, p_value = stats.ttest_ind(the_group, the_other_group)
    # print(n, m, t_stat, p_value)


    df_corr = df_clean #[df_clean['over_50'] == 1]
    df_corr = df_corr[df_corr['adj_joy_cnt'] > 0]
    # df_corr = df_corr[df_corr['Sum_of_surprise_Episodes'] > 0]

    est = smf.ols(formula="correct_learning_questions_percent ~  adj_joy_cnt * C(condition_stop)",
                  data=df_corr).fit()
    print(est.summary())

    x = df_corr['adj_joy_cnt'].values
    y = df_corr['correct_learning_questions_percent'].values
    print(len(x))
    plt.plot(x, y, 'bo')
    plt.show()