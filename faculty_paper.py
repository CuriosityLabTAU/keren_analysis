from analyze_data import *
from show_results import *

# first load the data
df = load_data()

# clean the data, get only relevant columns and information
df_clean, relevant_keys, stat_keys, stat_names = get_relevant_data(df)
df_clean = renormalize_entropies(df_clean, stat_keys)
print(df_clean.columns)

for faculty_column in ['faculty_1st']:
    # df_clean = df_clean[df_clean['condition_stop'] == 1]
    faculty_list = np.unique(df_clean[faculty_column].dropna().values)
    print('================== ', faculty_column, ' ==================')
    for the_things in stat_keys:
        the_key = the_things#relevant_keys.index(the_things)
        print('--------- ', the_things, '----------')

        # ==== one-side anova
        print ('*** ***')
        fac_data = []  # a list of numpy array of all the data from the faculty (per key)
        for fac1 in faculty_list:
            fac_data.append(df_clean[df_clean[faculty_column] == fac1][the_key].dropna().values)
            print(fac1, ' N: ', fac_data[-1].shape[0])
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