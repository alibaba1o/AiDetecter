from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
# def plot_dendrogram(model, **kwargs):
#     # Create linkage matrix and then plot the dendrogram
#     # create the counts of samples under each node
#     counts = np.zeros(model.children_.shape[0])
#     n_samples = len(model.labels_)
#     for i, merge in enumerate(model.children_):
#         current_count = 0
#         for child_idx in merge:
#             if child_idx < n_samples:
#                 current_count += 1  # leaf node
#             else:
#                 current_count += counts[child_idx - n_samples]
#         counts[i] = current_count
#
#     linkage_matrix = np.column_stack(
#         [model.children_, model.distances_, counts]
#     ).astype(float)
#
#     # Plot the corresponding dendrogram
#     dendrogram(linkage_matrix, **kwargs)

data = np.genfromtxt('update_clasters.csv', delimiter=',',
                     usecols=np.arange(1,13),skip_header=1,dtype=float,encoding='utf-8')
X = np.array(data)
scaler = MinMaxScaler()
X_dendro=scaler.fit_transform(X).T
X_dendro=linkage(X_dendro, method='average', metric='chebyshev')
#data = np.genfromtxt('update_clasters.csv', delimiter=',',usecols=np.arange(1),
#                     skip_header=1,dtype=float,encoding='utf-8')
#y = np.array(data)
#
# clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
# clustering= clustering.fit(X)

print(X_dendro.shape)
my_labels = ['word_count','char_count','sentence_count','avg_word_length','avg_sentence_length','unique_word_ratio','uppercase_ratio','syllable_ratio','punctuation_count','readabillity','peplexity','compressionRatio']
plt.figure(figsize=(15,7))
dendrogram(X,p=30,labels=my_labels,leaf_font_size=7,)

plt.title('Дендрограмма признаков')
# plot_dendrogram(clustering, truncate_mode="level", p=3)
plt.ylabel("Расстояние")
plt.show()
#print(kmeans.labels_)