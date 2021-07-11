
# from bingwang import X_test
from libsvm.svm import PRINT_STRING_FUN
from matplotlib import colors
from numpy.random import f
from pandas.io.formats.format import DataFrameFormatter
from scipy.sparse.construct import random
import seaborn
# from pylab import rcParams
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.sparse import data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

from scipy.interpolate import make_interp_spline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, LogisticRegression
import joblib
import scikitplot
# KNN算法


def KNN_infer():
    iris_data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris_data.data, iris_data.target, random_state=10)
    print(y_test)
    print('-'*40)
    trf = StandardScaler()
    x_train = trf.fit_transform(x_train)
    x_test = trf.transform(x_test)

    est = KNeighborsClassifier()
    ary = np.arange(2, 10)
    params = {'n_neighbors': ary}
    grid = GridSearchCV(est, param_grid=params, n_jobs=-1, cv=10)

    test_predict_label = grid.fit(x_train, y_train).decision_function(x_test)
    fpr, tpr, threadshold = roc_curve(y_test, test_predict_label)
    roc_auc = auc(fpr, tpr)

    y_predict = est.predict(x_test)
    # print()
    print('预测目标为：', y_predict)
    print("实际目标为：", y_test)
    print('预测结果：', y_predict == y_test)

    score = est.score(x_test, y_test)
    # print(score)
    print('最佳估计器', grid.best_estimator_)

    print('最佳参数：', grid.best_params_)
    print('最佳结果：', grid.best_score_)

    print('最佳交叉验证结果：', grid.cv_results_)
    plt.plot(fpr, tpr, lw=2, label='Roc curve (area =%0.2f)' % roc_auc)
    plt.show()


# 朴素贝叶斯算法


def nb_news():
    news = fetch_20newsgroups(subset='all')
    x_train, x_test, y_train, y_test = train_test_split(
        news.data, news.target, test_size=0.75, random_state=10)

    # 特征工程
    tranfer = TfidfVectorizer()
    x_train = tranfer.fit_transform(x_train)
    x_test = tranfer.transform(x_test)

    est = MultinomialNB()

    # est=GridSearchCV(est)
    test_predict_label = est.fit(x_train, y_train).decision_function(x_test)
    fpr, tpr, threadshold = roc_curve(y_test, test_predict_label)
    roc_auc = auc(fpr, tpr)

    y_predict = est.predict(x_test)
    # print('预测目标为：', y_predict)
    # print("实际目标为：", y_test)
    print('预测结果：', y_predict == y_test)

    score = est.score(x_test, y_test)
    print(score)
    plt.plot(fpr, tpr, lw=2, label='Roc curve (area =%0.2f)' % roc_auc)
    plt.show()


# 降维
def instacart_PAC():
    pass
# 决策树


def DecisionTree():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=22)
    est = DecisionTreeClassifier(criterion='entropy')

    # 特征工程标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    est.fit(x_train, y_train)
    y_predict = est.predict(x_test)
    score = est.score(x_test, y_test)
    print('预测结果：', y_predict == y_test)
    print('准确率：', score)
    pass


def TaiTanic():
    data = pd.read_csv('./taitanic.csv')
    x = data[['Pclass', 'Age', 'Sex']]
    x['Age'].fillna(x['Age'].mean(), inplace=True)
    y = data['Survived']
    x = x.to_dict(orient='records')
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    transfer = DictVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    est = DecisionTreeClassifier(criterion='entropy')
    est.fit(x_train, y_train)
    y_predict = est.predict(x_test)
    score = est.score(x_test, y_test)
    print('预测结果：', y_predict == y_test)
    print('准确率：', score)

# 随机森林


def RandomForesttemp():
    data = pd.read_csv('./taitanic.csv')
    x = data[['Pclass', 'Age', 'Sex']]
    x['Age'].fillna(x['Age'].mean(), inplace=True)
    y = data['Survived']
    x = x.to_dict(orient='records')

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    transfer = DictVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    est = RandomForestClassifier()
    param_dict = {'n_estimators': np.random.randint(
        150, 200, 10), 'max_depth': np.random.randint(20, 30, 10)}
    grid = GridSearchCV(est, param_grid=param_dict, cv=10)

    grid.fit(x_train, y_train)
    y_predict = grid.predict(x_test)

    print('验证结果：', y_predict == y_test)
    print('最佳估计器', grid.best_estimator_)

    print('最佳参数：', grid.best_params_)
    print('最佳结果：', grid.best_score_)

    print('最佳交叉验证结果：', grid.cv_results_)
    pass


def linear1():
    bst = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(
        bst.data, bst.target, random_state=22, test_size=0.8)
    # x_test = x_test.values
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    est = LinearRegression()
    est.fit(x_train, y_train)
    # joblib.dump(est, '正规方程.pkl')
    # est = joblib.load('正规方程.pkl')
    print('正规方程权重系数:', est.coef_)
    print('偏置为：', est.intercept_)
    y_predict = est.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print('均方误差为：', error)
    # plt.scatter(x_test, y_test)
    plt.plot(x_test, y_predict, linewidth=2)
    plt.show()
    print(bst)
    pass


def linear2():
    bst = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(
        bst.data, bst.target, random_state=22)

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    est = SGDRegressor(max_iter=10000, eta0=0.01,
                       learning_rate='adaptive', penalty='l2')
    est.fit(x_train, y_train)
    joblib.dump(est, '梯度下降.pkl')
    print('梯度下降权重系数:', est.coef_)
    print('偏置为：', est.intercept_)
    y_predict = est.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print('均方误差为：', error)
    pass

# 岭回归


def Ridge_test():
    bst = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(
        bst.data, bst.target, random_state=22)

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    est = Ridge(alpha=0.5)
    est.fit(x_train, y_train)
    joblib.dump(est, '岭回归.pkl')
    print('岭回归权重系数:', est.coef_)
    print('偏置为：', est.intercept_)
    y_predict = est.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print('均方误差为：', error)

    pass

    aisles = pd.read_csv(
        'D:\\Anaconda\\python\\two\\test1\\Market-Basket-Analysis-Instacart-master\\instacart-market-basket-analysis\\aisles.csv')
    order_products = pd.read_csv(
        'D:\\Anaconda\\python\\two\\test1\\Market-Basket-Analysis-Instacart-master\\instacart-market-basket-analysis\\order_products__prior.csv')
    orders = pd.read_csv(
        'D:\\Anaconda\\python\\two\\test1\\Market-Basket-Analysis-Instacart-master\\instacart-market-basket-analysis\\orders.csv')
    products = pd.read_csv(
        'D:\\Anaconda\\python\\two\\test1\\Market-Basket-Analysis-Instacart-master\\instacart-market-basket-analysis\\products.csv')
    # table1 = pd.merge(aisles, products, on=['aisle_id', 'aisle_id'])
    print(products)
    # table2=pd.merge(table1,order_products,on='product_id')
    # print(table1)
    pass


def multiply():
    boston_data = load_boston()
    # print(boston_data.DESCR)
    boston_data_frame = DataFrame(
        data=boston_data.data, columns=boston_data.feature_names)
    x = boston_data_frame[boston_data_frame.columns[0:12]]
    y = boston_data_frame[boston_data_frame.columns[12:14]]
    print(x)
    print(y)
    seaborn.heatmap(x.corr())
    # plt.show()

    # 去除高相关性
    abs_corr_matrix = x.abs()
    up_tri = abs_corr_matrix.where(
        np.triu(np.ones(abs_corr_matrix.shape), k=1).astype(np.bool_))
    # print(up_tri)
    corr_features = [
        column for column in up_tri.columns if any(up_tri[column] > 0.75)]
    # print(corr_features)
    # print(x.shape)
    x = x.drop(corr_features, axis=1)
    # print(x.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    est = LinearRegression()

    est.fit(x_train, y_train)
    y_predict = est.predict(x_test)
    m_s_e = mean_squared_error(y_test, y_predict)
    print('均方误差为：', m_s_e)
    print('预测估计：', len(y_predict))
    r2_s = r2_score(y_test, y_predict)
    print('R方系数为：', r2_s)
    print('平均绝对误差为：', mean_absolute_error(y_test, y_predict))
    score = est.score(x_test, y_test)
    print('准确率为：', score)
    grid = GridSearchCV(est, param_grid={}, cv=10)
    grid.fit(x_train, y_train)
    # print(grid.best_score_)
    # print(grid.)
    print(grid.best_estimator_)
    print(len(grid.predict(x_test)))
    print(grid.cv_results_)

    # K-cv 交叉验证计算误差
    mean_abs_error = list()
    K_f = KFold(n_splits=10, shuffle=True)
    for train, test in K_f.split(x):
        est.fit(np.array(x)[train], np.array(y)[train])
        y1_test = np.array(y)[test]
        y1_predict = est.predict(np.array(x)[test])
        tmp = mean_absolute_error(y1_test, y1_predict)
        mean_abs_error.append(tmp)
        # print(train, test)

    print("--------")
    print(np.mean(mean_abs_error))
    # print(np.array(x)[train], np.array(x)[test])
    print("*"*50)

    # error_sum = 0
    # print(K_f.split(x_train))
    # for train, test in K_f.split(x, y):
    #     train_frame = DataFrame(data=train)
    #     test_frame = DataFrame(data=test)
    #     x1_train, x1_test, y1_train, y1_test = train_test_split(
    #         train_frame, test_frame)
    #     est.fit(x1_train, y1_train)
    #     # x1_train, x1_test = x[train_index], x[test_index]
    #     # y1_train, y1_test = y[train_index], y[test_index]

    #     y1_predict = est.predict(x1_test)
    #     error_sum += mean_absolute_error(y1_test, y1_predict)
    # print(error_sum)
    # print(K_f.split(x))
    # for item1, item2 in K_f.split(x, y):
    #     print(item1)
    #     print(item2)
    #     print('-'*50)
    # est.fit()
    # for
    pass


def Lessso_RES():
    boston_data = load_boston()
    # print(boston_data.DESCR)
    boston_data_frame = DataFrame(
        data=boston_data.data, columns=boston_data.feature_names)
    x = boston_data_frame[boston_data_frame.columns[0:12]]
    y = boston_data_frame[boston_data_frame.columns[12:14]]
    # print(x)
    # print(y)
    # seaborn.heatmap(x.corr())
    # plt.show()

    # 去除高相关性
    abs_corr_matrix = x.abs()
    up_tri = abs_corr_matrix.where(
        np.triu(np.ones(abs_corr_matrix.shape), k=1).astype(np.bool_))
    # print(up_tri)
    corr_features = [
        column for column in up_tri.columns if any(up_tri[column] > 0.75)]
    # print(corr_features)
    # print(x.shape)
    x = x.drop(corr_features, axis=1)
    # print(x.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print(x_test.isnull().any())
    est = Lasso(alpha=0.001, normalize=True, max_iter=1000)
    est.fit(x_train, y_train)
    y_predict = est.predict(x_test)
    print('平均绝对误差：', mean_absolute_error(y_test, y_predict))
    print(x_test)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex='col', sharey='row')
    a, b, c, d, error = lasso_resgression_with_degree(
        x_train, y_train, x_test, y_test, 16, 0)
    ax1.scatter(a, b)
    ax1.plot(c, d)
    ax1.set_title('alpha=0')
    a, b, c, d, error = lasso_resgression_with_degree(
        x_train, y_train, x_test, y_test, 16, 0.001)
    ax2.scatter(a, b)
    ax2.plot(c, d)
    ax2.set_title('alpha=0.001')
    a, b, c, d, error = lasso_resgression_with_degree(
        x_train, y_train, x_test, y_test, 16, 0.1)
    ax3.scatter(a, b)
    ax3.plot(c, d)
    ax3.set_title('alpha=0.1')
    a, b, c, d, error = lasso_resgression_with_degree(
        x_train, y_train, x_test, y_test, 16, 1)
    ax4.scatter(a, b)
    ax4.plot(c, d)
    ax4.set_title('alpha=1')
    plt.show()
    pass


def lasso_resgression_with_degree(x_train, y_train, x_test, y_test, degree, alpha):
    mdl = make_pipeline(PolynomialFeatures(degree), Lasso(alpha=alpha))
    mdl.fit(x_train, y_train)

    x_test = DataFrame(data=x_test)
    x_test = x_test.sort_values(by=['CRIM'])

    # x_test.fillna(x_test['CRIM'].mean())
    print(x_test)
    print(x_test.isnull().any())
    y_test = DataFrame(data=y_test)
    y_test = y_test.loc[x_test.index]
    y_predict = mdl.predict(x_test)
    smooth_feature = np.linspace(
        np.min(x_test['CRIM'].tolist()), np.max(x_test['CRIM'].tolist()), 200)
    smooth_points = make_interp_spline(
        x_test['CRIM'].tolist(), y_predict)(smooth_feature)
    return x_test['CRIM'], y_test, smooth_feature, smooth_points, mean_absolute_error(y_test, y_predict)

    pass


def Log_RES():
    data = pd.read_csv('./taitanic.csv')
    titanic_data = data[['Survived', 'Pclass', 'Sex',
                         'Age', 'SibSp', 'Parch', 'Embarked', 'Fare']]
    # print(titanic_data.shape)
    # print(titanic_data.isnull().any())
    # print(titanic_data.isnull().sum())
    titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
    titanic_data['Embarked'].dropna(inplace=True)
    # print(titanic_data.dtypes)
    X = titanic_data[['Pclass', 'Sex',
                      'Age', 'SibSp', 'Parch', 'Embarked', 'Fare']]
    Y = titanic_data[['Survived']]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    # seaborn.distplot(x_train['Age'])

    # 标准化
    tranfer = StandardScaler()
    x_train[['Age']] = tranfer.fit_transform(x_train[['Age']])
    x_test[['Age']] = tranfer.transform(x_test[['Age']])

    x_train[['Fare']] = tranfer.fit_transform(x_train[['Fare']])
    x_test[['Fare']] = tranfer.transform(x_test[['Fare']])

    x_train['Sex'] = x_train['Sex'].map({'female': 1, 'male': 0})
    x_test['Sex'] = x_test['Sex'].map({'female': 1, 'male': 0})

    # 将Embarked进行编码转换
    embarked_encode = preprocessing.LabelEncoder()
    x_train[['Embarked']] = embarked_encode.fit_transform(
        x_train[['Embarked']])
    x_test[['Embarked']] = embarked_encode.transform(x_test[['Embarked']])

    del x_train['Pclass']
    del x_test['Pclass']
    x_train = x_train.values
    x_test = x_test.values
    y_train = y_train.values
    y_test = y_test.values
    # print(x_train.info()
    logistic_regression = LogisticRegression()
    logistic_regression.fit(x_train, y_train)

    y_predict = logistic_regression.predict(x_test)
    print('预测结果：', y_predict == y_test)
    # 学习曲线和sv曲线绘制
    print('准确率为：', logistic_regression.score(x_test, y_test))
    scikitplot.estimators.plot_learning_curve(
        logistic_regression, x_train, y_train)

    # 预测概率
    y_pre_prob = logistic_regression.predict_proba(x_test)
    # print(y_pre_prob)
    class_1_prob = list()
    for item in y_pre_prob:
        class_1_prob.append(item[1])
    # print(roc_auc_score(y_test, class_1_prob))
    # 绘制roc曲线
    scikitplot.metrics.plot_roc_curve(
        y_test, y_pre_prob, curves=['each_class'])
    # 绘制混肴矩阵
    scikitplot.metrics.plot_confusion_matrix(y_test, y_predict, normalize=True)
    plt.show()

    pass


def TDesion_tree():
    data = pd.read_csv('./taitanic.csv')
    titanic_data = data[['Survived', 'Pclass', 'Sex',
                         'Age', 'SibSp', 'Parch', 'Embarked', 'Fare']]
    # print(titanic_data.shape)
    # print(titanic_data.isnull().any())
    # print(titanic_data.isnull().sum())
    titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
    titanic_data['Embarked'].dropna(inplace=True)
    # print(titanic_data.dtypes)
    X = titanic_data[['Pclass', 'Sex',
                      'Age', 'SibSp', 'Parch', 'Embarked', 'Fare']]
    Y = titanic_data[['Survived']]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    # seaborn.distplot(x_train['Age'])

    # 标准化
    tranfer = StandardScaler()
    x_train[['Age']] = tranfer.fit_transform(x_train[['Age']])
    x_test[['Age']] = tranfer.transform(x_test[['Age']])

    x_train[['Fare']] = tranfer.fit_transform(x_train[['Fare']])
    x_test[['Fare']] = tranfer.transform(x_test[['Fare']])

    x_train['Sex'] = x_train['Sex'].map({'female': 1, 'male': 0})
    x_test['Sex'] = x_test['Sex'].map({'female': 1, 'male': 0})

    # 将Embarked进行编码转换
    embarked_encode = preprocessing.LabelEncoder()
    x_train[['Embarked']] = embarked_encode.fit_transform(
        x_train[['Embarked']])
    x_test[['Embarked']] = embarked_encode.transform(x_test[['Embarked']])

    del x_train['Pclass']
    del x_test['Pclass']
    x_train = x_train.values
    x_test = x_test.values
    y_train = y_train.values
    y_test = y_test.values

    est = DecisionTreeClassifier()
    est = GridSearchCV(est, param_grid={'max_depth': np.arange(3, 10)}, cv=10)
    est.fit(x_train, y_train)
    y_predict = est.predict(x_test)
    y_pre_prob = est.predict_proba(x_test)
    scikitplot.estimators.plot_learning_curve(est, x_train, y_train)
    scikitplot.metrics.plot_confusion_matrix(y_test, y_predict, normalize=True)

    class_1_prob = list()
    for item in y_pre_prob:
        class_1_prob.append(item[1])
    print('roc准确率：', roc_auc_score(y_test, class_1_prob))
    scikitplot.metrics.plot_roc_curve(
        y_test, y_pre_prob, curves=['each_class'])
    print('预测结果：', y_predict == y_test)
    # print('准确率：', est.score(x_test, y_test))
    print('最佳参数:', est.best_params_)
    print('最佳准确率：', est.best_score_)
    plt.show()
    pass
# if __name__ == '__main__':
# KNN_infer()
# nb_news()
# DecisionTree()
# TaiTanic()
# print(np.random.randint(100, 150, 10))
# RandomForesttemp()
# linear1()
# linear2()
# Ridge_test()
# K_means_test()
# multiply()
# Lessso_RES()


# y = np.array([1.53E+03, 5.92E+02, 2.04E+02,
#              7.24E+01, 2.72E+01, 1.10E+01, 4.70E+00])
# x_smooth = np.linspace(x.min(), x.max(), 300)
# y_smooth = make_interp_spline(x, y, x_smooth)
# plt.plot(x_smooth, y_smooth)
# plt.show()
# Log_RES()
# TDesion_tree()
