import pandas
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'

df = pd.read_csv('datas/embedding_output_1k.csv')

#print(df['embedding'][0])
#print(type(df['embedding'][0]))  # str

# 把str转化成矩阵
df['embedding_vec'] = df['embedding'].apply(ast.literal_eval)

#print(len(df['embedding_vec'][0]))
#print(type(df['embedding_vec'][0]))  # list

# T-SNE 可以将高维度的数据映射到 2D 或 3D 的空间中, 以便我们可以直观地观察和理解数据的结构
if df['embedding_vec'].apply(len).nunique() == 1:
    matrix = np.vstack(df['embedding_vec'].values)

    # 创建一个T-SNE模型
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    # 使用t-sne模型降维，得到一个二维的
    matrix_2d = tsne.fit_transform(matrix)

    print(matrix_2d)

    # 可视化, 根据不同的颜色来区分不同的评分
    colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]

    x = matrix_2d[:, 0]
    y = matrix_2d[:, 1]

    # 评分是从1开始的，（减1是因为评分是从1开始的，而颜色索引是从0开始的）获取对应的颜色索引
    colors_indices = df.Score.values - 1

    color_map = matplotlib.colors.ListedColormap(colors)

    # 画出一个散点图
    plt.scatter(x=x, y=y, c=colors_indices, cmap=color_map, alpha=0.3)

    plt.title('使用T-SNE降维后的亚马逊评论')

    plt.show()