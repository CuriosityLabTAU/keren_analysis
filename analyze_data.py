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