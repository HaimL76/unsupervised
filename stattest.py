import os

import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils import replace_extension, csv_to_dict

# קריאת הדאטהסט
file_path = r"ds\schizophrenia_dataset.csv"
df = pd.read_csv(file_path)

trans_file = replace_extension(file_path, '.trans')

column_map: dict

if os.path.exists(trans_file):
    column_map = csv_to_dict(trans_file)

    df.rename(columns=column_map, inplace=True)

# 1. מבחן t: האם יש הבדל בגיל בין חולי סכיזופרניה לבין אלו שלא?
group1 = df[df["Tanı"] == 1]["Yaş"]  # קבוצת חולי סכיזופרניה
group2 = df[df["Tanı"] == 0]["Yaş"]  # קבוצת הלא-חולים
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"מבחן t להשוואת גילאים בין חולים ללא חולים: T={t_stat:.4f}, p={p_value:.4f}")

# 2. מבחן חי-בריבוע: האם היסטוריה משפחתית קשורה לסיכון לחלות?
contingency_table = pd.crosstab(df["Ailede_Şizofreni_Öyküsü"], df["Tanı"])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"מבחן חי-בריבוע להיסטוריה משפחתית: Χ²={chi2:.4f}, p={p:.4f}")

# 3. רגרסיה לוגיסטית: אילו גורמים מנבאים סכיזופרניה?
logit_model = smf.logit("Tanı ~ Yaş + Cinsiyet + Eğitim_Seviyesi + Gelir_Düzeyi + Ailede_Şizofreni_Öyküsü + Madde_Kullanımı + Stres_Faktörleri", data=df).fit()
print(logit_model.summary())

# 4. מבחן t: האם יש הבדל בציון תפקוד (GAF) בין אלו שנוטלים תרופות לאלו שלא?
group1 = df[df["İlaç_Uyumu"] == 1]["GAF_Skoru"]  # קבוצה של נוטלי תרופות
group2 = df[df["İlaç_Uyumu"] == 0]["GAF_Skoru"]  # קבוצה של לא-נוטלים
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"מבחן t לציון GAF: T={t_stat:.4f}, p={p_value:.4f}")

# 5. מבחן קורלציה: האם יש קשר בין גורמי לחץ לסימפטומים חיוביים?
correlation, p_value = stats.pearsonr(df["Stres_Faktörleri"], df["Pozitif_Semptom_Skoru"])
print(f"מבחן קורלציה בין גורמי לחץ לסימפטומים חיוביים: r={correlation:.4f}, p={p_value:.4f}")