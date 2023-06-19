from classify import knn

group, labels = knn.createDataSet()
print(knn.classify0([0,0], group, labels,3))
# print(group)
# print(labels)