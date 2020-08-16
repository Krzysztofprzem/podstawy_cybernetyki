import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np


def plotClustersCenters(ax, centersMemberships, centersXY):
    centersX = np.ndarray.tolist(centersXY[0])
    centersY = np.ndarray.tolist(centersXY[1])
    ax.scatter(centersX, centersY, c='black', s=40)
    ax.scatter(centersX, centersY, c=centersMemberships, s=20, cmap='gist_rainbow')


def plot_clusters_memberships_neural_network(ax, string, memberships, xy):
    x = np.ndarray.tolist(xy[0])
    y = np.ndarray.tolist(xy[1])
    ax.grid()
    ax.scatter(x, y, c=memberships, s=20, cmap='gist_rainbow')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.axis('equal')
    ax.set_title(string)


def load_files(path):
    lines = [line.rstrip('\n') for line in open(path)]
    amount = len(lines)
    points = np.zeros((amount, 2))
    points_member = np.zeros(amount)

    for i in range(len(lines)):
        split = lines[i].split()
        points[i, 0] = float(split[0])
        points[i, 1] = float(split[1])
        points_member[i] = int(split[2])
    return points, points_member



def create_datasets(tr_coun, va_count, te_count):
    dataset = [line.rstrip('\n') for line in open('kmeans.txt')]
    np.random.shuffle(dataset)

    valid = open('valid.txt', "w")
    train = open('train.txt', "w")
    test = open('test.txt', "w")
    valid_count = 0
    train_count = 0
    test_count = 0
    # dataset = []
    #
    # for i in range(len(x)):
    #     dataset.append([x[i], y[i], memberships[i]])
    np.random.shuffle(dataset)

    for i in range(len(dataset)):
        split = dataset[i].split()
        file = valid
        case = np.random.randint(0, 3)
        if case == 0 and train_count < tr_coun:
            file = train
            train_count += 1
        elif case == 1 and valid_count < va_count:
            file = valid
            valid_count += 1
        elif case == 2 and test_count < te_count:
            file = test
            test_count += 1
        else:
            if valid_count < va_count:
                file = valid
                valid_count += 1
            elif train_count < tr_coun:
                file = train
                train_count += 1
            else:
                file = test
                test_count += 1

        file.write(str(split[0]))
        file.write(" ")
        file.write(str(split[1]))
        file.write(" ")
        file.write(str(split[2]))
        file.write("\n")

    valid.close()
    train.close()
    test.close()



def save_results(xy, memberships):
    x = np.ndarray.tolist(xy[0])
    y = np.ndarray.tolist(xy[1])
    neuron = open('neuron.txt', "w")
    for i in range(len(x)):
        neuron.write(str(int(x[i])))
        neuron.write(" ")
        neuron.write(str(int(y[i])))
        neuron.write(" ")
        neuron.write(str(int(memberships[i])))
        neuron.write("\n")
    neuron.close()


create_datasets(3250, 750, 1000)

valid_points, valid_member = load_files('valid.txt')
train_points, train_member = load_files('train.txt')
test_points, test_member = load_files('test.txt')
kmeans_points, kmeans_members = load_files('kmeans.txt')
kmeansC_points, kmeansC_members = load_files('kmeansC.txt')

valid_points = valid_points/1000000
train_points = train_points/1000000
test_points = test_points/1000000


model = keras.Sequential([
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='tanh'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(15, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_points, train_member, epochs=20, validation_data=(valid_points, valid_member))

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.grid()
plt.legend()
plt.show()



test_loss, test_acc = model.evaluate(test_points, test_member, verbose=2)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


model.save_weights('model.h5')


#model.load_weights('model.h5')


memberships_model = model.predict(test_points)
memberships_model = (np.argmax(memberships_model, axis=1))

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(13, 5))
plot_clusters_memberships_neural_network(ax1, "train", train_member, train_points.transpose()*1000000)
plot_clusters_memberships_neural_network(ax2, "valid", valid_member, valid_points.transpose()*1000000)
plot_clusters_memberships_neural_network(ax3, "test", memberships_model, test_points.transpose()*1000000)
plot_clusters_memberships_neural_network(ax4, "k-means", kmeans_members, kmeans_points.transpose())
plotClustersCenters(ax4, kmeansC_members, kmeansC_points.transpose())
plt.show()

save_results(test_points.transpose()*1000000, memberships_model)
