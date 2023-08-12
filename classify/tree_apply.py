from . import trees, trees_plot
import pandas as pd
import numpy as np

def classifyHomeData(filePath):
    homeData =  pd.read_csv(filePath)
    y = homeData["Transported"]
    # choose 10 features
    features = ["PassengerId", "HomePlanet", "CryoSleep", "Cabin", "Destination",
                "Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa"]
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
    trees_plot.createPlot(myTree)
    # print(myTree)


if __name__ == "__main__":
    classifyHomeData()