import numpy as np
import matplotlib.pyplot as plt
import json

def plotClustersInit(x, y, centersX, centersY):
    plt.grid()
    plt.scatter(x, y, c='b')
    plt.scatter(centersX, centersY, c='black', s=40)
    plt.scatter(centersX, centersY, c='magenta', s=20)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))
    plt.axis('equal')
    plt.show()



def plotClustersEpoch(memberships, x, y, centersMemberships, centersX, centersY, oldCentersX, oldCentersY):
    # print(centersX)
    # print(oldCentersX)
    plt.grid()
    plt.scatter(x, y, c=memberships, s=20, cmap='gist_rainbow')
    plt.scatter(oldCentersX, oldCentersY, c='black', s=40)
    plt.scatter(centersX, centersY, c='black', s=40)
    plt.scatter(centersX, centersY, c=centersMemberships, s=20, cmap='gist_rainbow')
    for i in range(len(oldCentersX)):
        plt.plot([oldCentersX, centersX], [oldCentersY, centersY], 'r', 100)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))
    plt.axis('equal')
    plt.show()



def plotClustersMemberships(memberships, x, y, centersMemberships, centersX, centersY):
    plt.grid()
    plt.scatter(x, y, c=memberships, s=20, cmap='gist_rainbow')
    plt.scatter(centersX, centersY, c='black', s=40)
    plt.scatter(centersX, centersY, c=centersMemberships, s=20, cmap='gist_rainbow')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))
    plt.axis('equal')
    plt.show()




def KMeans(x, y, C, centersX, centersY, epsilon=0.1, show=True):
    N = len(x)
    iterations = True
    # centersX = []
    # centersY = []
    # for i in range(C):
    #     newX = np.random.randint(min(x), max(x))
    #     newY = np.random.randint(min(y), max(y))
    #     centersX.append(float(newX))
    #     centersY.append(float(newY))

    # group all points by their current memberships to cluster centers
    if show:
        memberships = []
        for j in range(N):
            # print("point: ", j, "\tx:", x[j], "y: ", y[j])
            nearest = 0
            dx = x[j] - centersX[0]
            dy = y[j] - centersY[0]
            min_distance = np.sqrt(dx * dx + dy * dy)
            # print("distance to center: 0: ", min_distance, "\tx:", centersX[0], "y: ", centersY[0])
            for k in range(1, C):
                dx = x[j] - centersX[k]
                dy = y[j] - centersY[k]
                distance = np.sqrt(dx * dx + dy * dy)
                # print("distance to center: ", k, ": ", distance, "\tx:", centersX[k], "y: ", centersY[k])
                if distance < min_distance:
                    min_distance = distance
                    nearest = k
            # print("membership of point: ", j, ": ", nearest)
            memberships.append(nearest)

        # group all points by their current memberships to cluster centers
        centersMemberships = []
        for j in range(C):
            nearest = 0
            dx = x[0] - centersX[j]
            dy = y[0] - centersY[j]
            min_distance = np.sqrt(dx * dx + dy * dy)
            for k in range(1, N):
                dx = x[k] - centersX[j]
                dy = y[k] - centersY[j]
                distance = np.sqrt(dx * dx + dy * dy)
                if distance < min_distance:
                    min_distance = distance
                    nearest = memberships[k]
            centersMemberships.append(nearest)

        plotClustersMemberships(memberships, x, y, centersMemberships, centersX, centersY)

    new_centersX = [0]*C
    new_centersY = [0]*C

    while iterations:
        memberships = []
        # group all points by their current memberships to cluster centers
        for j in range(N):
            nearest = 0
            dx = x[j] - centersX[0]
            dy = y[j] - centersY[0]
            min_distance = np.sqrt(dx * dx + dy * dy)
            for k in range(1, C):
                dx = x[j]-centersX[k]
                dy = y[j]-centersY[k]
                distance = np.sqrt(dx*dx + dy*dy)
                if distance < min_distance:
                    min_distance = distance
                    nearest = k
            memberships.append(nearest)

        # compute new positions of cluster centers
        for center in range(C):
            sumx = 0
            sumy = 0
            amount = 0
            for mem in range(len(memberships)):
                if memberships[mem] == center:
                    sumx += x[mem]
                    sumy += y[mem]
                    amount += 1

            if amount != 0:
                new_centersX[center] = sumx/amount
                new_centersY[center] = sumy/amount


        # check if every cluster center moved distance is smaller than epsilon
        for i in range(C):
            dx = new_centersX[i] - centersX[i]
            dy = new_centersY[i] - centersY[i]
            distance = np.sqrt(dx*dx + dy*dy)
            # print(distance)
            if distance > epsilon:
                break
            else:
                if i == C-1:
                    iterations = False

        # group all points by their current memberships to cluster centers
        memberships = []
        for j in range(N):
            # print("point: ", j, "\tx:", x[j], "y: ", y[j])
            nearest = 0
            dx = x[j] - new_centersX[0]
            dy = y[j] - new_centersY[0]
            min_distance = np.sqrt(dx * dx + dy * dy)
            # print("distance to center: 0: ", min_distance, "\tx:", centersX[0], "y: ", centersY[0])
            for k in range(1, C):
                dx = x[j]-new_centersX[k]
                dy = y[j]-new_centersY[k]
                distance = np.sqrt(dx*dx + dy*dy)
                # print("distance to center: ", k, ": ", distance, "\tx:", centersX[k], "y: ", centersY[k])
                if distance < min_distance:
                    min_distance = distance
                    nearest = k
            # print("membership of point: ", j, ": ", nearest)
            memberships.append(nearest)

        # group all points by their current memberships to cluster centers
        centersMemberships = []
        for j in range(C):
            nearest = 0
            dx = x[0] - new_centersX[j]
            dy = y[0] - new_centersY[j]
            min_distance = np.sqrt(dx * dx + dy * dy)
            for k in range(1, N):
                dx = x[k] - new_centersX[j]
                dy = y[k] - new_centersY[j]
                distance = np.sqrt(dx * dx + dy * dy)
                if distance < min_distance:
                    min_distance = distance
                    nearest = memberships[k]
            centersMemberships.append(nearest)

        if show:
            plotClustersEpoch(memberships, x, y, centersMemberships, new_centersX, new_centersY, centersX, centersY)

        centersX = new_centersX
        centersY = new_centersY

    return x, y, memberships, centersX, centersY, centersMemberships





def load_data(choose=False, path='S1.txt', path_init='init.txt'):
    lines = [line.rstrip('\n') for line in open(path)]

    x = []
    y = []
    for i in range(len(lines)):
        split = lines[i].split()
        x.append(float(split[0]))
        y.append(float(split[1]))


    generate = False
    if choose:
        print("min x: ", min(x), "\tmax x: ", max(x), "\tmin y: ", min(y), "\tmax y: ", max(y))
        print("0: Load initial clusters centers from file\n1: Generate random positions")
        generate = int(input())

    centersX = []
    centersY = []
    C = 0
    if generate == 1 and choose:
        print("How many clusters centers?")
        C = int(input())
        for i in range(C):
            newX = np.random.randint(min(x), max(x))
            newY = np.random.randint(min(y), max(y))
            centersX.append(float(newX))
            centersY.append(float(newY))
    elif generate == 0:
        lines = [line.rstrip('\n') for line in open(path_init)]
        for i in range(len(lines)):
            split = lines[i].split()
            centersX.append(float(split[0]) + np.random.randint(-100000, 100000))
            centersY.append(float(split[1]) + np.random.randint(-100000, 100000))
            C = len(centersX)

    return x, y, centersX, centersY, C



def create_datasets(x, y, memberships, centersX, centersY, centersMemberships):
    kmeans = open('kmeans.txt', "w")
    for i in range(len(x)):
        kmeans.write(str(int(x[i])))
        kmeans.write(" ")
        kmeans.write(str(int(y[i])))
        kmeans.write(" ")
        kmeans.write(str(memberships[i]))
        kmeans.write("\n")
    kmeans.close()

    kmeansCenters = open('kmeansC.txt', "w")
    for i in range(len(centersX)):
        kmeansCenters.write(str(centersX[i]))
        kmeansCenters.write(" ")
        kmeansCenters.write(str(centersY[i]))
        kmeansCenters.write(" ")
        kmeansCenters.write(str(centersMemberships[i]))
        kmeansCenters.write("\n")
    kmeansCenters.close()





if __name__ == "__main__":
    x, y, centersX, centersY, C = load_data()
    plotClustersInit(x, y, centersX, centersY)
    x2, y2, memberships, centersX, centersY, centersMemberships = KMeans(x, y, C, centersX, centersY, 0.1, False)
    plotClustersMemberships(memberships, x2, y2, centersMemberships, centersX, centersY)
    create_datasets(x, y, memberships, centersX, centersY, centersMemberships)