import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def load_data():
    lines = [line.rstrip('\n') for line in open("S1.txt")]
    x = []
    y = []
    for i in range(len(lines)):
        split = lines[i].split()
        x.append(float(split[0]))
        y.append(float(split[1]))
    return x, y

def load_compare_data(filepath):
    lines = [line.rstrip('\n') for line in open(filepath)]
    x = []
    y = []
    memberships = []
    for i in range(len(lines)):
        split = lines[i].split()
        x.append(float(split[0]))
        y.append(float(split[1]))
        memberships.append(int(split[2]))
    return x, y, memberships


def plotClustersMemberships(ax, string, memberships, x, y):
    ax.grid()
    ax.scatter(x, y, c=memberships, s=20, cmap='gist_rainbow')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.axis('equal')
    ax.set_title(string)

def plotClustersCenters(ax, centersMemberships, centersX, centersY):
    ax.scatter(centersX, centersY, c='black', s=40)
    ax.scatter(centersX, centersY, c=centersMemberships, s=20, cmap='gist_rainbow')

def compare(fuzzyMemberships, fuzzyX, fuzzyY, kmeansMemberships, kmeansX, kmeansY):
    counter = 0
    all = len(fuzzyX)
    for i in range(len(fuzzyX)):
        if fuzzyX[i] == kmeansX[i]:
            if fuzzyY[i] == kmeansY[i]:
                # counter += 1
                if fuzzyMemberships[i] == kmeansMemberships[i]:
                    counter += 1
    return counter*100/all



fuzzyX, fuzzyY = load_data()
xDomain = ctrl.Antecedent(np.arange(0,1000000,10000),'xDomain')

xDomain['very_small'] = fuzz.trimf(xDomain.universe,[0,150000,240000])
xDomain['small'] = fuzz.trimf(xDomain.universe,[220000,300000,440000])
xDomain['average'] = fuzz.trimf(xDomain.universe,[400000,450000,600000])
xDomain['big'] = fuzz.trimf(xDomain.universe,[530000,640000,740000])
xDomain['very_big'] = fuzz.trimf(xDomain.universe,[730000,820000,1000000])

xDomain.view()
plt.grid()
plt.show()

yDomain=ctrl.Antecedent(np.arange(0,1000000,10000),'yDomain')

yDomain['very_small'] = fuzz.trimf(yDomain.universe,[0, 120000,240000])
yDomain['small'] = fuzz.trimf(yDomain.universe,[220000,400000,500000])
yDomain['average'] = fuzz.trimf(yDomain.universe,[450000,600000,670000])
yDomain['big'] = fuzz.trimf(yDomain.universe,[640000,740000,800000])
yDomain['very_big'] = fuzz.trimf(yDomain.universe,[700000,820000,1000000])

yDomain.view()
plt.grid()
plt.show()



cluster=ctrl.Consequent(np.arange(-1,16,1),'cluster')
for i in range(16):
    cluster[str(i)] = fuzz.trimf(cluster.universe, [i-1, i, i+1])

    
cluster.view()
plt.grid()
plt.show()
    
rules = []

rules.append(ctrl.Rule(antecedent=((xDomain['very_small'] & yDomain['very_small'])), consequent=cluster['14']))
rules.append(ctrl.Rule(antecedent=((xDomain['very_small'] & yDomain['small'])), consequent=cluster['8']))
rules.append(ctrl.Rule(antecedent=((xDomain['very_small'] & yDomain['average'])), consequent=cluster['3']))
rules.append(ctrl.Rule(antecedent=((xDomain['very_small'] & yDomain['big'])), consequent=cluster['0']))
rules.append(ctrl.Rule(antecedent=((xDomain['very_small'] & yDomain['very_big'])), consequent=cluster['0']))

rules.append(ctrl.Rule(antecedent=((xDomain['small'] & yDomain['very_small'])), consequent=cluster['12']))
rules.append(ctrl.Rule(antecedent=((xDomain['small'] & yDomain['small'])), consequent=cluster['9']))
rules.append(ctrl.Rule(antecedent=((xDomain['small'] & yDomain['average'])), consequent=cluster['3']))
rules.append(ctrl.Rule(antecedent=((xDomain['small'] & yDomain['big'])), consequent=cluster['4']))
rules.append(ctrl.Rule(antecedent=((xDomain['small'] & yDomain['very_big'])), consequent=cluster['0']))

rules.append(ctrl.Rule(antecedent=((xDomain['average'] & yDomain['very_small'])), consequent=cluster['13']))
rules.append(ctrl.Rule(antecedent=((xDomain['average'] & yDomain['small'])), consequent=cluster['9']))
rules.append(ctrl.Rule(antecedent=((xDomain['average'] & yDomain['average'])), consequent=cluster['4']))
rules.append(ctrl.Rule(antecedent=((xDomain['average'] & yDomain['big'])), consequent=cluster['2']))
rules.append(ctrl.Rule(antecedent=((xDomain['average'] & yDomain['very_big'])), consequent=cluster['1']))

rules.append(ctrl.Rule(antecedent=((xDomain['big'] & yDomain['very_small'])), consequent=cluster['13']))
rules.append(ctrl.Rule(antecedent=((xDomain['big'] & yDomain['small'])), consequent=cluster['10']))
rules.append(ctrl.Rule(antecedent=((xDomain['big'] & yDomain['average'])), consequent=cluster['5']))
rules.append(ctrl.Rule(antecedent=((xDomain['big'] & yDomain['big'])), consequent=cluster['2']))
rules.append(ctrl.Rule(antecedent=((xDomain['big'] & yDomain['very_big'])), consequent=cluster['2']))

rules.append(ctrl.Rule(antecedent=((xDomain['very_big'] & yDomain['very_small'])), consequent=cluster['14']))
rules.append(ctrl.Rule(antecedent=((xDomain['very_big'] & yDomain['small'])), consequent=cluster['11']))
rules.append(ctrl.Rule(antecedent=((xDomain['very_big'] & yDomain['average'])), consequent=cluster['7']))
rules.append(ctrl.Rule(antecedent=((xDomain['very_big'] & yDomain['big'])), consequent=cluster['6']))
rules.append(ctrl.Rule(antecedent=((xDomain['very_big'] & yDomain['very_big'])), consequent=cluster['2']))





clustering_ctrl=ctrl.ControlSystem(rules)
clustering = ctrl.ControlSystemSimulation(clustering_ctrl)


# print(clustering)

fuzzyMemberships=fuzzyX.copy()
for i in range(len(fuzzyX)):
    clustering.input['xDomain'] = fuzzyX[i]
    clustering.input['yDomain'] = fuzzyY[i]
    clustering.compute()
    fuzzyMemberships[i] = int(clustering.output['cluster'])

# print(z)
# print(compare(z,x,y))


kmeansX, kmeansY, kmeansMemberships = load_compare_data("kmeans.txt")
kmeansCX, kmeansCY, kmeansMembershipsC = load_compare_data("kmeansC.txt")
neuronX, neuronY, neuronMemberships = load_compare_data("neuron.txt")



print("Similarity to kmeans:", compare(fuzzyMemberships, fuzzyX, fuzzyY, kmeansMemberships, kmeansX, kmeansY), "%")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 5))
# plotClustersMemberships(z, x, y)
plotClustersMemberships(ax1, "k-means", kmeansMemberships, kmeansX, kmeansY)
plotClustersCenters(ax1, kmeansMembershipsC, kmeansCX, kmeansCY)
plotClustersMemberships(ax2, "neural network", neuronMemberships, neuronX, neuronY)
plotClustersMemberships(ax3, "fuzzy", fuzzyMemberships, fuzzyX, fuzzyY)
plt.show()




# print(compare(z,x,y))