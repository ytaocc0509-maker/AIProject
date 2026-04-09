import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'

df = pd.read_csv('datas/embedding_output_1k.csv')

# 把str转化成矩阵
df['embedding_vec'] = df['embedding'].apply(ast.literal_eval)

# T-SNE 可以将高维度的数据映射到 2D 或 3D 的空间中, 以便我们可以直观地观察和理解数据的结构
#  nunique()，返回不同值的个数。[1,1,1,1,1,1]
if df['embedding_vec'].apply(len).nunique() == 1:
    # 把Embedding之后的向量空间变成矩阵
    matrix = np.vstack(df['embedding_vec'].values)

    # 初始化KMeans对象
    km = KMeans(3, init='k-means++', random_state=43, n_init=10)
    km.fit(matrix)

    df['Kmeans_Label'] = km.labels_  # 把所有类别存放的新的字段： 0，1，2

    # 创建一个T-SNE模型
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)

    # 使用t-sne模型降维，得到一个二维的
    matrix_2d = tsne.fit_transform(matrix)

    print(matrix_2d)

    # 可视化, 根据不同的颜色来区分不同的类别
    colors = ["red", "green", "blue"]

    x = matrix_2d[:, 0]
    y = matrix_2d[:, 1]

    # 根据类别的值，来得到不同的颜色
    colors_indices = df['Kmeans_Label'].values

    color_map = matplotlib.colors.ListedColormap(colors)

    # 画出一个散点图
    plt.scatter(x=x, y=y, c=colors_indices, cmap=color_map)

    plt.title('使用聚类和T-SNE降维后的亚马逊评论')

    plt.show()
