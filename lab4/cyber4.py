import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
from skimage import measure



def computeSOM():
    points = []
    lines = [line.rstrip('\n') for line in open('breast.txt')]
    for i in range(len(lines)):
        split = lines[i].split()
        point = []
        for j in range(len(split)):
            point.append(float(split[j]))
        points.append(point)

    maxs = 0
    for i in range(len(points)):
        maxL = max(points[i])
        if maxL > maxs:
            maxs = maxL

    for i in range(len(points)):
        for j in range(len(points[i])):
            points[i][j]=points[i][j]/maxs

    som = MiniSom(30, 30, 9, sigma=0.5, learning_rate=0.7, random_seed=3)

    print("Training...")
    som.train_batch(points, 3000, verbose=True)  # random training
    print("\n...ready!")

    bone()
    pcolor(som.distance_map().T, cmap='gist_heat')
    colorbar()
    for i, x in enumerate(points):
        w = som.winner(x)
    plot(w[0], w[1])
    show()

    plt.figure(figsize=(30, 30))
    # Plotting the response for each pattern in the iris dataset
    plt.pcolor(som.distance_map().T, cmap='gist_heat')  # plotting the distance map as background
    #plt.colorbar()

    for cnt, xx in enumerate(points):
        w = som.winner(xx)  # getting the winner
        # palce a marker on the winning position for the sample xx
        plt.plot(w[0]+.5, w[1]+.5, 'D', markerfacecolor='None',
                 markeredgecolor='C1', markersize=12, markeredgewidth=2)
    plt.axis([0, 30, 0, 30])
    plt.show()


    plt.figure(figsize=(30, 30))
    # Plotting the response for each pattern in the iris dataset
    dists = som.distance_map().T.copy()
    for i in range(len(dists)):
        for j in range(len(dists[i])):
            if dists[i][j]<0.5:
                dists[i][j] = 1
            else:
                dists[i][j] = 0


    for i in range(len(dists)):
        for j in range(len(dists[i])):
            dists[i][0] = 0
            dists[i][len(dists)-1] = 0
            dists[0][j] = 0
            dists[len(dists[i])-1][j] = 0

    plt.pcolor(dists, cmap='gist_heat')  # plotting the distance map as background

    all_labels = measure.label(dists)
    blobs_labels = measure.label(dists, background=0)


    plt.pcolor(blobs_labels, cmap='gist_heat')  # plotting the distance map as background
    plt.plot()

    memberships = []
    for cnt, xx in enumerate(points):
        w = som.winner(xx)  # getting the winner
        memberships.append(all_labels[w[0],w[1]])

    return memberships



def computeDistance(points, centers, j, k):
    dxi = [0 for i in range(len(points))]
    distSum = 0
    for i in range(len(dxi)):
        dxi[i] = points[i][j] - centers[i][k]
        distSum += dxi[i] ** 2
    return np.sqrt(distSum)

def computeMemberships(points, centers, N, C):
    memberships = []
    for j in range(N):
        nearest = 0
        min_distance = computeDistance(points, centers, j, 0)
        for k in range(1, C):
            distance = computeDistance(points, centers, j, k)
            if distance < min_distance:
                min_distance = distance
                nearest = k
        memberships.append(nearest)
    return memberships


def computeCentersMemberships(points, centers, N, C, memberships):
    centersMemberships = []
    for j in range(C):
        nearest = 0
        min_distance = computeDistance(points, centers, 0, j)
        for k in range(1, N):
            distance = computeDistance(points, centers, k, j)
            if distance < min_distance:
                min_distance = distance
                nearest = memberships[k]
        centersMemberships.append(nearest)
    return centersMemberships

def computeNewCenters(points, memberships, C):
    new_centers = [[0] * C for i in range(len(points))]
    for center in range(C):
        sums = [0 for i in range(len(points))]
        amount = 0
        for mem in range(len(memberships)):
            if memberships[mem] == center:
                for i in range(len(points)):
                    sums[i] += points[i][mem]
                amount += 1

        if amount != 0:
            for i in range(len(points)):
                new_centers[i][center] = sums[i] / amount

    return new_centers


def checkCondition(centers, new_centers, C, epsilon):
    for i in range(C):
        distance = computeDistance(new_centers, centers, i, i)
        # print(distance)
        if distance > epsilon:
            return True
        else:
            if i == C - 1:
                return False

def KMeans(points, C, centers, epsilon=0.1, show=True):
    N = len(points[0])
    iterations = True

    # if show:
    #     memberships = computeMemberships(points, centers, N, C)
    #     # group all points by their current memberships to cluster centers
    #     centersMemberships = computeCentersMemberships(points, centers, N, C, memberships)
    #
    #     # plotClustersMemberships(memberships, x, y, centersMemberships, centersX, centersY)

    new_centers = [[0]*C for i in range(len(points))]


    while iterations:
        memberships = computeMemberships(points, centers, N, C)
        # compute new positions of cluster centers
        new_centers = computeNewCenters(points, memberships, C)
        # check if every cluster center moved distance is smaller than epsilon
        iterations = checkCondition(centers, new_centers, C, epsilon)

        centers = new_centers

    # group all points by their current memberships to cluster centers
    memberships = computeMemberships(points, centers, N, C)
    # group all points by their current memberships to cluster centers
    centersMemberships = computeCentersMemberships(points, centers, N, C, memberships)

        # if show:
        #     plotClustersEpoch(memberships, x, y, centersMemberships, new_centersX, new_centersY, centersX, centersY)

        #centers = new_centers

    return points, memberships, centers, centersMemberships


def plot_breast2(points, memberships):
    fig = plt.figure()
    axs = []
    for i in range(1,4):
        axs.append(fig.add_subplot(130+i, projection='3d'))

    for i in range(3):
        axs[i].scatter(points[i*3], points[i*3+1], points[i*3+2], c=memberships, s=20, cmap='gist_rainbow')
        axs[i].set_xlabel(str(i*3))
        axs[i].set_ylabel(str(i*3+1))
        axs[i].set_zlabel(str(i*3+2))
    plt.show()


def plot_breast(points, memberships, centers, centersMemberships):
    from itertools import permutations
    dimensions = [i for i in range(9)]
    perm = permutations(dimensions, 3)
    good = []
    for permutation in list(perm):
        if set(permutation) not in good:
            good.append(set(permutation))
    # print(len(list(good)))
    # exit()



    figs = [plt.figure(), plt.figure()]
    for k in range(2):
        # fig = plt.figure()
        axs = []
        for i in range(1, int(len(good)/2+1)):
            axs.append(figs[k].add_subplot(6, 7, i, projection='3d'))

        for i in range(int(len(good)/2)):
            one, two, three = list(good[k*int(len(good)/2) + i])[0], list(good[k*int(len(good)/2) + i])[1], list(good[k*int(len(good)/2) + i])[2]
            axs[i].scatter(points[one], points[two], points[three], c=memberships, s=20, cmap='gist_rainbow')
            axs[i].scatter(centers[one], centers[two], centers[three], c='black', s=100)
            axs[i].scatter(centers[one], centers[two], centers[three], c=centersMemberships, s=50, cmap='gist_rainbow')
            axs[i].set_xlabel(str(one))
            axs[i].set_ylabel(str(two))
            axs[i].set_zlabel(str(three))
    plt.show()



def parallelCoordinates(points, membershipsSOM):
    colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y', 'k', 'w']

    for i in range(len(points[0])): # wymiar i
        x = []
        y = []
        for j in range(len(points)):
            x.append(j)
            y.append(points[j][i])  # konkretny punkt w wymiarze i

        plt.plot(x, y, c=colors[membershipsSOM[i]])

    plt.show()


def load_data(path='breast.txt'):
    lines = [line.rstrip('\n') for line in open(path)]

    points = [[] for i in range(len(lines[0].split()))]
    for i in range(len(lines)):
        split = lines[i].split()
        for j in range(len(split)):
            points[j].append(float(split[j]))


    centers = [[] for i in range(len(points))]
    C = 2
    for i in range(C):
        for j in range(len(centers)):
            centers[j].append(np.random.randint(min(points[j]), max(points[j])))

    return points, centers, C



membershipsSOM = computeSOM()




points, centers, C = load_data()
# points, memberships, centers, centersMemberships = KMeans(points, C, centers, epsilon=0.1, show=True)
# plot_breast(points, memberships, centers, centersMemberships)
plot_breast2(points, membershipsSOM)
parallelCoordinates(points, membershipsSOM)