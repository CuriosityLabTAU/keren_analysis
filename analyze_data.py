from get_data import *
from scipy import stats
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.decomposition import factor_analysis
import matplotlib.pyplot as plt
import seaborn as sns


# df_clean, relevant_keys, stat_keys = get_data()
# faculty_list, n_faculty = get_faculty(df_clean, relevant_keys)
# results = get_results(df_clean, relevant_keys, stat_keys, faculty_list, n_faculty)


def analyze_faculty(df_clean, relevant_keys, stat_keys, faculty_list, results):
    for the_things in stat_keys:
        the_key = relevant_keys[the_things]
        print('--------- ', the_key, '----------')
        fac_data = []
        for fac1 in faculty_list:
            fac_data.append(df_clean[df_clean[relevant_keys[3]] == fac1][the_key].dropna().values)
        print(stats.kruskal(fac_data[0], fac_data[1], fac_data[2],
                       fac_data[3], fac_data[4], fac_data[5],
                       fac_data[6], fac_data[7], fac_data[8]))
        #     n_fac1_data = fac1_data.shape[0]
        #     for fac2 in faculty_list:
        #         fac2_data = df_clean[df_clean[relevant_keys[3]] == fac2][the_key].dropna()
        #         n_fac2_data = fac2_data.shape[0]
        #         z_stat, p_val = stats.ranksums(fac1_data, fac2_data)
        #
        #         if p_val < 0.05:
        #             print(the_key, n_fac1_data, n_fac2_data,
        #                   get_fac(fac1), get_fac(fac2), z_stat, p_val,
        #                   results['faculty']['mean'][get_fac(fac1)][the_key],
        #                   results['faculty']['mean'][get_fac(fac2)][the_key])


def analyze_multi_regression(df_clean, relevant_keys, stat_keys):
    free_exp_stats = [relevant_keys[k] for k in [-7, -3, -2]]
    label_stat = [relevant_keys[k] for k in [-4]]
    regression_stats = [relevant_keys[k] for k in [-7, -3, -2, -4]]
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

    fig, ax = plt.subplots()
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()


def analyze_regression(df_clean, relevant_keys, stat_keys):
    for the_things1 in stat_keys:
        the_key1 = relevant_keys[the_things1]
        for the_things2 in [-7, -4, -3, -2, -1]:
            the_key2 = relevant_keys[the_things2]
            if the_key1 == the_key2:
                keys_df = df_clean[[the_key1]].dropna()
                data_key1 = keys_df[:][the_key1]
                z_stat, p_val = stats.ranksums(data_key1, data_key1)
                print(the_key1, the_key2, data_key1.shape, data_key1.shape, z_stat, p_val)
            else:
                keys_df = df_clean[[the_key1, the_key2]].dropna()
                # keys_df.to_csv(the_key1 + the_key2 + '.csv')
                print(keys_df.head())
                data_key1 = keys_df[the_key1]
                data_key2 = keys_df[the_key2]
                slope, intercept, r_value, p_value, std_err = stats.linregress(data_key1.values, data_key2.values)
                print(the_key1, the_key2, data_key1.shape, data_key2.shape, r_value, p_value)
                if p_value < 0.05:
                    sns.regplot(x=data_key1, y=data_key2)
                    plt.show()


def analyze_factor_analysis(df_clean, relevant_keys, stat_keys):
    df_fa = df_clean[relevant_keys].dropna()

    fa = factor_analysis()
    fa.fit(df_fa.values)

