import pandas as pd
import scipy.stats as stats


def run_chi2(clusters: list):
    all_categories = set()
    counts_list = []

    for cluster in clusters:
        s = pd.Series(cluster)
        counts = s.value_counts()
        counts_list.append(counts)
        all_categories.update(counts.index)

    all_categories = sorted(all_categories)

    table = []
    for counts in counts_list:
        row = [counts.get(cat, 0) for cat in all_categories]
        table.append(row)

    return stats.chi2_contingency(table)
