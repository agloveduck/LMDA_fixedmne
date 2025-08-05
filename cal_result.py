import numpy as np
import pandas as pd

# 每个被试的 Best Test Accuracy（单位：%）/mean_accs
subjects = [f"A{str(i).zfill(2)}" for i in range(1, 10)]
# best_accs = [
#     85.06944, 64.93056, 89.93056, 78.12500, 65.62500,
#     61.80556, 92.70833, 82.63889, 85.41667
# ]
best_accs = [
    82.04861, 62.46528, 88.22917, 76.28472, 61.28472,
    56.25000, 90.03472, 79.06250, 83.68056
]

# 计算平均准确率和标准差
mean_accuracy = np.mean(best_accs)
std_accuracy = np.std(best_accs, ddof=1)

# 计算 Kappa 值（4类分类，随机准确率是0.25）
chance_level = 0.25
kappas = [(acc/100 - chance_level) / (1 - chance_level) for acc in best_accs]
mean_kappa = np.mean(kappas)
std_kappa = np.std(kappas, ddof=1)

# 保留一位小数
mean_accuracy = round(mean_accuracy, 1)
std_accuracy = round(std_accuracy, 1)
mean_kappa = round(mean_kappa, 2)
std_kappa = round(std_kappa, 2)

# 打印结果
print(f"Average Accuracy: {mean_accuracy} ± {std_accuracy} %")
print(f"Average Kappa: {mean_kappa} ± {std_kappa}")

# 也可以输出每个被试的详细信息
summary_df = pd.DataFrame({
    "Subject": subjects,
    "Best Accuracy (%)": [round(x, 2) for x in best_accs],
    "Kappa": [round(k, 3) for k in kappas]
})
print(summary_df)
