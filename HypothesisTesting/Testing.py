import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp, mannwhitneyu

# T-test
# t-test for proof that 6 own classes are representative on validation set
class_val = pd.read_csv('Class_metrics_error_subset_val.csv')
method_val = pd.read_csv('Method_metrics_error_subset_val.csv')

res_class = ttest_1samp(class_val['Error'], popmean=0.25)
res_method = ttest_1samp(method_val['Error'], popmean=0.25)
print("Representative t-test")
print(f"P-value for class: {res_class.pvalue}")
print(f"P-value for method: {res_method.pvalue}")

# t-test for proof that train and test samples are equivalent
class_train = pd.read_csv('Class_metrics_error_subset_train.csv')
method_train = pd.read_csv('Method_metrics_error_subset_train.csv')

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
print("For class")
statistic, p_value = mannwhitneyu(class_train['Error'], class_val['Error'])
print(f"Mann-Whitney U statistic: {statistic}")
print(f"P-value: {p_value}")
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Samples are not equivalent.")
else:
    print("Fail to reject the null hypothesis: Samples are equivalent.")

print("For method")
statistic, p_value = mannwhitneyu(method_train['Error'], method_val['Error'])
print(f"Mann-Whitney U statistic: {statistic}")
print(f"P-value: {p_value}")
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Samples are not equivalent.")
else:
    print("Fail to reject the null hypothesis: Samples are equivalent.")
