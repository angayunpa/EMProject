import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp, mannwhitneyu

# T-test
# t-test for proof that 6 own classes are representative on validation set
class_val = pd.read_csv('SammonErrors/Class_metrics_error_subset_val.csv')
method_val = pd.read_csv('SammonErrors/Method_metrics_error_subset_val.csv')

res_class = ttest_1samp(class_val['Error'], popmean=0.25)
res_method = ttest_1samp(method_val['Error'], popmean=0.25)
print("Representative t-test")
print(f"P-value for class: {res_class.pvalue}")
print(f"P-value for method: {res_method.pvalue}")

# t-test for proof that train and test samples are equivalent
class_train = pd.read_csv('SammonErrors/Class_metrics_error_subset_train.csv')
method_train = pd.read_csv('SammonErrors/Method_metrics_error_subset_train.csv')

print("for class")
statistic, p_value = ttest_ind(class_train['Error'], class_val['Error'])
print(f"T-statistic: {statistic}")
print(f"P-value: {p_value}")
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Samples are not equivalent.")
else:
    print("Fail to reject the null hypothesis: Samples are equivalent.")

print("for method")
statistic, p_value = ttest_ind(method_train['Error'], method_val['Error'])
print(f"T-statistic: {statistic}")
print(f"P-value: {p_value}")
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Samples are not equivalent.")
else:
    print("Fail to reject the null hypothesis: Samples are equivalent.")



# Mann-Whitney test
Mann_Whitney_p_value = pd.DataFrame({
    'n_comp': range(2, 21),
    'class_train': 0,
    'method_train': 0
})

class_pso = pd.read_csv('PSO/PSO_Sammon_class_error_train.csv')
class_ga = pd.read_csv('GA/GA_Sammon_class_error_train.csv')

method_pso = pd.read_csv('PSO/PSO_Sammon_method_error_train.csv')
method_ga = pd.read_csv('GA/GA_Sammon_method_error_train.csv')

for i in range(2, len(class_pso.columns)):
    # Perform Mann-Whitney U test
    class_pso_nth = class_pso[class_pso.columns[i]].to_list()
    class_ga_nth = class_ga[class_ga.columns[i]].to_list()
    statistic, p_value = mannwhitneyu(class_pso_nth, class_ga_nth)
    
    # FOR CLASS
    print(f"n_components = {i}")
    print(f"CLASS")
    print(f"Mann-Whitney U statistic: {statistic}")
    print(f"P-value: {p_value}")
    Mann_Whitney_p_value.iloc[i-2, 1] = p_value

    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference between the distributions.\n")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference between the distributions.\n")

    # FOR METHOD
    method_pso_nth = method_pso[method_pso.columns[i]].to_list()
    method_ga_nth = method_ga[method_ga.columns[i]].to_list()
    statistic, p_value = mannwhitneyu(method_pso_nth, method_ga_nth)
    print(f"METHOD")
    print(f"Mann-Whitney U statistic: {statistic}")
    print(f"P-value: {p_value}")
    Mann_Whitney_p_value.iloc[i-2, 2] = p_value

    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference between the distributions.\n")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference between the distributions.\n")
Mann_Whitney_p_value.to_csv("Mann_Whitney_error.csv", sep='\t')
