# from analyze_data import *
from show_results import *
from pylab import *
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import statsmodels.api as sm


# first load the data
df = load_data()

# clean the data, get only relevant columns and information
df_clean, relevant_keys, stat_keys, stat_names = get_relevant_data(df)


def stars(p):
   if p < 0.0001:
       return "****"
   elif (p < 0.001):
       return "***"
   elif (p < 0.01):
       return "**"
   elif (p < 0.05):
       return "*"
   else:
       return "-"


def condition_plot(a, b, x, a_str, b_str, x_str, df_, x_name):
    x_stop_0 = x[b == 0].dropna()
    x_stop_1 = x[b == 1].dropna()
    x_priming_0 = x[a == 0].dropna()
    x_priming_1 = x[a == 1].dropna()

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
        ab_lbl = ['NO-STOP/NO-PRIMING', 'NO-STOP/PRIMING', 'STOP/NO-PRIMING', 'STOP/PRIMING']
        x_data = []
        x_data.append(x[(a == 0) & (b == 0)])
        x_data.append(x[(a == 1) & (b == 0)])
        x_data.append(x[(a == 0) & (b == 1)])
        x_data.append(x[(a == 1) & (b == 1)])
        n_x = 4
        x_cond = np.array([1, 2, 3, 4])

    fig = figure()
    ax = fig.add_subplot(111)

    bp = ax.boxplot(x_data)

    params = {
        'axes.labelsize': 24,
        'text.fontsize': 24,
        'legend.fontsize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'text.usetex': True,
        'figure.figsize': [20, 10]
    }

    rcParams.update(params)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', length=0)

    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.2)

    ax.set_xticklabels(ab_lbl)
    ax.set_ylabel('$%s$' % x_name)

    # y_max = np.max(np.concatenate(x_data))
    # y_min = np.min(np.concatenate(x_data))
    #
    # ax.annotate("", xy=(1.5, y_max), xycoords='data',
    #             xytext=(3.5, y_max), textcoords='data',
    #             arrowprops=dict(arrowstyle="-", ec='#111111',
    #                             connectionstyle="bar,fraction=0.2"))
    #
    #
    # ax.text(2.5, y_max + abs(y_max - y_min) * 0.1, stars(stop_p),
    #         horizontalalignment='center',
    #         verticalalignment='center')
    # #
    #
    # print('Stop: No: %2.2f \\pm  %2.2f, N=%d, Yes: %2.2f \\pm %2.2f, N=%d' %
    #       (np.mean(x_stop_0), np.std(x_stop_0), len(x_stop_0),
    #        np.mean(x_stop_1), np.std(x_stop_1), len(x_stop_1)))
    # print('Priming: No: %2.2f \\pm  %2.2f, N=%d, Yes: %2.2f \\pm %2.2f, N=%d' %
    #       (np.mean(x_priming_0), np.std(x_priming_0), len(x_priming_0),
    #        np.mean(x_priming_1), np.std(x_priming_1), len(x_priming_1)))
    # print()
    #
    # stop_stat, stop_p = stats.ttest_ind(x_stop_0, x_stop_1)
    # priming_stat, priming_p = stats.ttest_ind(x_priming_0, x_priming_1)
    #
    #
    #
    # x_avg = np.zeros([n_x])
    # x_std = np.zeros([n_x])
    # for i_d, x_d in enumerate(x_data):
    #     x_avg[i_d] = np.mean(x_d)
    #     x_std[i_d] = np.std(x_d) / np.sqrt(len(x_d))
    #
    # data = x_avg
    # xlocations = x_cond
    # error = x_std
    # labels = ab_lbl
    #
    # # df_['ab'] = df_.apply(lambda row: row[a_str] + row[b_str] * 2, axis=1)
    # # df_na = df_[[x_str, 'ab']].dropna()
    # # mc = MultiComparison(df_na[x_str], df_na['ab'])
    # # result = mc.tukeyhsd()
    # # print(result.summary())
    # # print(mc.groupsunique)
    #
    # xlocations = np.array(range(len(data))) + 0.5
    # width = 0.5
    # the_linewidth = 10
    # bar(xlocations, data, yerr=error, width=width, ecolor='black')
    #
    #
    # # if stop_p < 0.05:
    # #     plot([xlocations[0] + 0.25, xlocations[1] + 0.25], [max(data)+max(error) * 1.2, max(data)+max(error) * 1.2],
    # #          '-', color='black', linewidth=the_linewidth)
    # #     plot([xlocations[2] + 0.25, xlocations[3] + 0.25], [max(data) + max(error) * 1.2, max(data) + max(error) * 1.2],
    # #          '-', color='black', linewidth=the_linewidth)
    # #     plot([(xlocations[0] + 0.25 + xlocations[1] + 0.25) / 2.0,
    # #           (xlocations[0] + 0.25 + xlocations[1] + 0.25) / 2.0,],
    # #          [max(data) + max(error) * 1.2, max(data) + max(error) * 1.7],
    # #          '-', color='black', linewidth=the_linewidth)
    # #     plot([(xlocations[2] + 0.25 + xlocations[3] + 0.25) / 2.0,
    # #           (xlocations[2] + 0.25 + xlocations[3] + 0.25) / 2.0, ],
    # #          [max(data) + max(error) * 1.2, max(data) + max(error) * 1.7],
    # #          '-', color='black', linewidth=the_linewidth)
    # #     plot([(xlocations[0] + 0.25 + xlocations[1] + 0.25) / 2.0,
    # #           (xlocations[2] + 0.25 + xlocations[3] + 0.25) / 2.0, ],
    # #          [max(data) + max(error) * 1.7, max(data) + max(error) * 1.7],
    # #          '-', color='black', linewidth=the_linewidth)
    # #     if stop_p < 0.001:
    # #         text(np.mean(xlocations) + 0.25, max(data) + max(error) * 2.0, '***')
    # #     elif stop_p < 0.01:
    # #         text(np.mean(xlocations) + 0.25, max(data) + max(error) * 2.0, '**')
    # #     elif stop_p < 0.05:
    # #         text(np.mean(xlocations) + 0.25, max(data) + max(error) * 2.0, '*')
    # #
    # # if priming_p < 0.05:
    # #     plot([xlocations[0] + 0.35, xlocations[2] + 0.15], [max(data) + max(error) * 1.2, max(data) + max(error) * 1.2],
    # #          '-', color='black', linewidth=the_linewidth)
    # #     plot([xlocations[1] + 0.35, xlocations[3] + 0.15], [max(data) + max(error) * 1.5, max(data) + max(error) * 1.5],
    # #          '-', color='black', linewidth=the_linewidth)
    # #     plot([(xlocations[0] + 0.25 + xlocations[2] + 0.25) / 2.0,
    # #           (xlocations[0] + 0.25 + xlocations[2] + 0.25) / 2.0, ],
    # #          [max(data) + max(error) * 1.2, max(data) + max(error) * 2.0],
    # #          '-', color='black', linewidth=the_linewidth)
    # #     plot([(xlocations[1] + 0.25 + xlocations[3] + 0.25) / 2.0,
    # #           (xlocations[1] + 0.25 + xlocations[3] + 0.25) / 2.0, ],
    # #          [max(data) + max(error) * 1.5, max(data) + max(error) * 2.0],
    # #          '-', color='black', linewidth=the_linewidth)
    # #     plot([(xlocations[0] + 0.25 + xlocations[2] + 0.25) / 2.0,
    # #           (xlocations[1] + 0.25 + xlocations[3] + 0.25) / 2.0, ],
    # #          [max(data) + max(error) * 2.0, max(data) + max(error) * 2.0],
    # #          '-', color='black', linewidth=the_linewidth)
    # #     if priming_p < 0.001:
    # #         text(np.mean(xlocations) + 0.25, max(data) + max(error) * 2.3, '***')
    # #     elif priming_p < 0.01:
    # #         text(np.mean(xlocations) + 0.25, max(data) + max(error) * 2.3, '**')
    # #     elif priming_p < 0.05:
    # #         text(np.mean(xlocations) + 0.25, max(data) + max(error) * 2.3, '*')
    #
    # # yticks(range(0, 8))
    # # xticks(xlocations + width / 2, labels)
    # plt.xticks([], [])
    #
    # xlim(0, xlocations[-1] + width * 2)
    # title_str = '%s, p_stop=%2.6f, p_priming=%2.6f' % (x_str, stop_p, priming_p)
    # # title(title_str)
    # # gca().get_xaxis().tick_bottom()
    # # gca().get_yaxis().tick_left()
    # # axis([xlocations[0], xlocations[-1] + 0.5, 0, max(data) + max(error) * 4.0])
    # # y_label = '$%s$' % x_name
    # # ylabel(y_label, usetex=True, fontsize=48)
    # plt.tick_params(axis='both', which='major', labelsize=48)
    # show()
    #
    # # print(x_avg, x_std)
    # # plt.bar(x_cond, x_avg)
    # # plt.errorbar(x_cond, x_avg, yerr=x_std, fmt='.', ecolor='black')
    # # plt.xticks(x_cond, ab_lbl)
    # # plt.title(x_str)
    # # plt.axis([0,5, 0, 1.0])
    plt.show()


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


df_clean = renormalize_entropies(df_clean, stat_keys)

analyze_conditions(df_clean, stat_keys, stat_names)