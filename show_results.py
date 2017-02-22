# Create bars for each thing, x-axis faculty, y-axis mean, error-bars, and * for statistical significance.
from get_data import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def label_diff(ax, i,j,text,X,Y):
    x = (X[i]+X[j])/2
    y = 1.1*max(Y[i], Y[j])
    dx = abs(X[i]-X[j])

    props = {'connectionstyle':'bar','arrowstyle':'-',\
                 'shrinkA':20,'shrinkB':20,'lw':2}
    ax.annotate(text, xy=(X[i],y+7), zorder=10)
    ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)


def show_faculty(df_clean, relevant_keys, stat_keys, faculty_list):
    for the_things in stat_keys:
        the_key = relevant_keys[the_things]
        fac_means = []
        for fac1 in faculty_list:
            fac_means.append(df_clean[df_clean[relevant_keys[3]] == fac1][the_key].mean())
        a = np.argsort(np.array(fac_means))
        fa = faculty_list[a]

        significant = []
        for if1, fac1 in enumerate(fa):
            fac1_data = df_clean[df_clean[relevant_keys[3]] == fac1][the_key].dropna().values
            for if2, fac2 in enumerate(fa):
                fac2_data = df_clean[df_clean[relevant_keys[3]] == fac2][the_key].dropna()
                n_fac2_data = fac2_data.shape[0]
                z_stat, p_val = stats.ranksums(fac1_data, fac2_data)
                if p_val < 0.05:
                    significant.append([if1, if2, p_val])
        print(significant)
        ax = sns.barplot(x=relevant_keys[3], y=the_key, data=df_clean, order=fa)

        # Keren:
        props = {'connectionstyle': 'bar', 'arrowstyle': '-', \
                 'shrinkA': 20, 'shrinkB': 20, 'lw': 2}
        ax.annotate('*', xy=(5.5,7), zorder=10)
        ax.annotate('', xy=(5, 5), xytext=(6, 5), arrowprops=props)

        f_name = [faculty_name[x] for x in a]
        ax.set(xticklabels=f_name, title=the_key)
        plt.show()