
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
myfont = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=14) # 解决不显示中文的问题
sns.set(font=myfont.get_name())
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
color = sns.color_palette()
from scipy.stats import norm
import numpy as np
import pandas as pd
from scipy import stats

# visualization
# histogram
def plot_histogram(data):
    sns.distplot(data['低能△IL'])
    # histogram and normal probability plot
    sns.distplot(data['能比'], fit=norm)
    sns.distplot(np.log(data['能比']), fit=norm)
    fig = plt.figure()
    res = stats.probplot(data['能比'], plot=plt)

    # skewness and kurtosis
    print("Skewness: %f" % data['低能△IL'].skew()) # Skewness: 0.680515
    print("Kurtosis: %f" % data['低能△IL'].kurtosis()) # Kurtosis: -0.345318

# relationshap with ash and low energy gap by scatter
def ash_scatter(input_x, input_y, str_x, str_y):
    data = pd.concat([input_x, input_y], axis=1)
    data.plot.scatter(x = str_x, y = str_y)
# ash_scatter(data['低能△IL'], data['灰分'], '灰分', '低能△IL')

# box plot ash / K value
def ash_boxplot(input_x, input_y, str_x, str_y):
    data = pd.concat([input_x, input_y], axis=1)
    f, ax = plt.subplots(figsize = (8, 6))
    fig = sns.boxplot(x = str_x, y = str_y, data=data)
    fig.axis(ymin=min(input_y), ymax = max(input_y))
# ash_boxplot(data['灰分'], data['高能△IH'], '灰分', '高能△IH')

# Correlation matrix(heatmap style)
def ash_corr(data):
    k = len(data.columns)
    corrmat = data.corr() # 默认Pearson相关系数，"kendall"表示Kendall Tau相关系数，"spearman"spearman相关系数
    # cols = corrmat.nlargest(k, '灰分')['灰分'].index
    # cm = np.corrcoef(data[cols].values.T) # Return Pearson correlation coefficients.
    sns.heatmap(corrmat, cbar = True, annot=True, square=False, fmt=".2f", annot_kws={'size':10})
# ash_corr(data)

# 多变量散点图
def ash_pairplot(data):
    sns.pairplot(data, size = 1.0)
# ash_pairplot(data[['灰分','低能△IL', '高能△IH', '低能比%', '高能比%', '能比']])
# ash_pairplot(data[['灰分', '原点距离', '固定点距离', '衰减比']])

# Outliers
# Standardizing data
# ash_scaled = StandardScaler().fit_transform(First_Ash_Monolayer_Test_rawdata['灰分'][:, np.newaxis])
# low_ash_range = ash_scaled[ash_scaled[:, 0].argsort()][:10]
# print('outer range (low) of the distribution:', low_ash_range)
# high_ash_range = ash_scaled[ash_scaled[:, 0].argsort()][-10:]
# print('outer range (high) of the distribution:', high_ash_range)

# ash ParallelPlot
def ash_parallelplot(all_data):
    summary = all_data.describe() # count mean std min 25% 50% 75% max
    # '灰分', 'K值', '低能IL0', '高能IH0', '低能IL', '高能IH', '低能△IL', '高能△IH', '低能比%',
    #        '高能比%', '原点距离', '固定点距离', '能比', '衰减比', '能减比'
    minash = summary.iloc[3, 0]
    maxash = summary.iloc[7, 0]
    nrows = len(all_data.index)
    # '低能IL0', '高能IH0', '低能IL', '高能IH', '低能△IL', '高能△IH'
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(nrows):
        # plot rows of data as if they were series data
        dataRow = all_data.iloc[i, 8:15]
        labelColor = (all_data.iloc[i, 0] - minash)/(maxash - minash) # 序号越高的，灰分越大，对应的颜色越深
        ax.plot(dataRow, color = plt.cm.RdYlBu(labelColor),alpha = 0.5)
    data_plot = all_data.iloc[0, 8:15]
    # ax.set_xticklabels(data_plot.columns, rotation = 45, fontsize = 'small')
    ax.set_xlabel("Attribute Index")
    ax.set_ylabel(("Attribute Values"))
    # ax.legend(loc='best',frameon=False )
    plt.show()
# ash_parallelplot(data)
#  '低能比%','高能比%', '原点距离', '固定点距离', '能比', '衰减比', '能减比' [8:15]

# 比较穿透后的低能IL、高能IH的差异，初步猜测这部分差异来自于厚度，因此用单层、双层、三层的数据共同来看
'''
H_Monolayer_Test = First_Ash_Monolayer_Test[['灰分','低能IL0','高能IH0','低能IL',	'高能IH','低能△IL','高能△IH',
                                                       '低能比%','高能比%',	'原点距离','能比']]
H_Monolayer_Test['num'] = 1

First_Ash_Double_Test = pd.read_excel(first_ash_text, sheet_name="双层试验") # (141, 11)
First_Ash_Double_Test = First_Ash_Double_Test.drop(0)
H_Double_Test = First_Ash_Double_Test[['灰分','低能IL0','高能IH0','低能IL',	'高能IH','低能△IL','高能△IH',
                                                       '低能比%','高能比%',	'原点距离',	'能比']]# (117, 11)
H_Double_Test['num'] = 2

First_Ash_Three_Test = pd.read_excel(first_ash_text, sheet_name="三层试验 ")
First_Ash_Three_Test = First_Ash_Three_Test.drop(0)
H_Three_tier_Test = First_Ash_Three_Test[['灰分','低能IL0','高能IH0','低能IL',	'高能IH','低能△IL','高能△IH',
                                                       '低能比%','高能比%',	'原点距离',	'能比']]# (42, 11)
H_Three_tier_Test['num'] = 3
data_test = pd.concat([H_Monolayer_Test, H_Double_Test, H_Three_tier_Test], axis=0, ignore_index=True)
data_test = data_test.sort_values(by = ['灰分','num'], axis=0, ascending=True)


ash_std = data_test[['低能IL',	'高能IH','低能△IL','高能△IH']].groupby(data_test['灰分']).std()
ash_std.describe()
ash_std = ash_std.rename(columns = {'低能IL':'低能IL_std', '高能IH':'高能IH_std', '低能△IL':'低能△IL_std', '高能△IH':'高能△IH_std'})


# 这里不需要选择平均值建模，可进行均值分析
ash_feature = pd.merge(data_test[['灰分', '低能IL0', '高能IH0', '低能IL', '高能IH',
       '低能△IL', '高能△IH', '低能比%', '高能比%', '原点距离', '能比', 'num']],
                       ash_std[['低能IL_std', '高能IH_std', '低能△IL_std','高能△IH_std']],
                       left_on= data_test['灰分'], right_on=ash_std.index, how='left')
ash_feature = ash_feature.drop('key_0', axis=1)
ash_feature['distance'] = np.sqrt(ash_feature['低能IL0'] ** 2 + ash_feature['高能IH0'] ** 2 +
                                  ash_feature['低能IL'] ** 2 + ash_feature['高能IH'] ** 2)/10000
ash_feature.corr()
# ash_corr(ash_feature)

# 仿照iris绘制图

ash_iris = np.array(ash_feature[['灰分', '低能IL0', '高能IH0', '低能IL', '高能IH', 'num']])
ash_iris_1 = ash_iris[ash_iris[:, 5] == 1.0]
ash_iris_2 = ash_iris[ash_iris[:, 5] == 2.0]
ash_iris_3 = ash_iris[ash_iris[:, 5] == 3.0]
ash_iris_1_split = np.hsplit(ash_iris_1[:, 1:5], 4)
ash_iris_2_split = np.hsplit(ash_iris_2[:, 1:5], 4)
ash_iris_3_split = np.hsplit(ash_iris_3[:, 1:5], 4)


I_1 = {'低能IL0':ash_iris_1_split[0], '高能IH0':ash_iris_1_split[1], '低能IL':ash_iris_1_split[2], '高能IH':ash_iris_1_split[3]}
I_2 = {'低能IL0':ash_iris_2_split[0], '高能IH0':ash_iris_2_split[1],'低能IL':ash_iris_2_split[2], '高能IH':ash_iris_2_split[3]}
I_3 = {'低能IL0':ash_iris_3_split[0], '高能IH0':ash_iris_3_split[1], '低能IL':ash_iris_3_split[2], '高能IH':ash_iris_3_split[3]}
size = 5
I_1_color = 'b'
I_2_color = 'g'
I_3_color = 'r'
label_text = ['低能IL0', '高能IH0', '低能IL', '高能IH']
plt.figure()
plt.suptitle('I Set(blue:单层, green:双层, red:三层)', fontsize = 30)
for i in range(0,4):
    for j in range(0,4):
        plt.subplot(4, 4, i * 4 + j + 1)

        if i == j:
            print(i * 4 + j + 1)

            plt.xticks([])
            plt.yticks([])
            plt.text(0.1, 0.4, label_text[i], size = 18)
        else:
            plt.scatter(ash_iris_1_split[j], ash_iris_1_split[i], c = I_1_color, s = size)
            plt.scatter(ash_iris_2_split[j], ash_iris_2_split[i], c = I_2_color, s = size)
            plt.scatter(ash_iris_3_split[j], ash_iris_3_split[i], c = I_3_color, s = size)
'''
