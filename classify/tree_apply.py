from . import trees
import pandas as pd
import numpy as np

def classifyHomeData(filePath):
    homeData =  pd.read_csv(filePath)
    y = homeData["SalePrice"]
    # choose 10 features
    features = ["MSSubClass", "MSZoning", "LotFrontage", "LandSlope", "LotShape","YearBuilt", "Heating", "1stFlrSF", "2ndFlrSF", "FullBath", "HalfBath"]
    X = homeData[features]

    y_list = y.values.tolist()
    x_list = X.values.tolist()
    n_x_list = np.array(x_list)
    n_y_list = np.array(y_list)
    n_data = np.c_[n_x_list, n_y_list.T]
    dataSet = n_data.tolist()
    labels = features

    # print(" ".join(str(i) for i in dataSet))
    featLabels = []
    myTree = trees.createTree(dataSet, labels, featLabels)

    print(myTree)


if __name__ == "__main__":
    classifyHomeData()