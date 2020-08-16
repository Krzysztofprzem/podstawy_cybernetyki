from keras.models import Sequential
from keras.layers import Dense
import scipy.io.wavfile as wave
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np



org = wave.read("org.wav")
short = wave.read("short2.wav")
long = wave.read("long2.wav")


print(org)

print(len(org))

org = np.array(org[1], dtype=float)/(65536/2)
short = np.array(short[1], dtype=float)/(65536/2)
long = np.array(long[1], dtype=float)/(65536/2)


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(13, 5))
time = [i for i in range(55500, 55500+1000)]
ax1.set_title("orig")
ax1.plot(time, org[55500:55500+1000])
ax1.grid()
ax1.set_xlabel("t")
ax1.set_ylabel("sample")
ax3.set_title("short")
ax3.plot(time, short[55500:55500+1000])
ax3.grid()
ax3.set_xlabel("t")
ax3.set_ylabel("sample")
ax5.set_title("long")
ax5.plot(time, long[55500:55500+1000])
ax5.grid()
ax5.set_xlabel("t")
ax5.set_ylabel("sample")



print(len(org))

num = 1000

timestep_x = []
timestep_y = []

time = [i for i in range(50000, 60000)]
for t in time:
    timestep_y.append(org[t])
    timestep_x.append(org[t-num:t])

train_x = np.array(timestep_x)
train_y = np.array(timestep_y)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=num))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss="mean_squared_error",
              optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_x, train_y, epochs=50)




time = [i for i in range(55500, 55500+1000)]
for t in time:
    short[t+1] = model.predict(np.array([short[t-num:t]]))

for t in time:
    long[t+1] = model.predict(np.array([long[t-num:t]]))



# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 5))

ax2.set_title("orig")
ax2.plot(time, org[55500:55500+1000])
ax2.grid()
ax2.set_xlabel("t")
ax2.set_ylabel("sample")
ax4.set_title("filled short")
ax4.plot(time, short[55500:55500+1000])
ax4.grid()
ax4.set_xlabel("t")
ax4.set_ylabel("sample")
ax6.set_title("filled long")
ax6.plot(time, long[55500:55500+1000])
ax6.grid()
ax6.set_xlabel("t")
ax6.set_ylabel("sample")

plt.show()



plt.plot(org[55500:55500+1000]-short[55500:55500+1000])
# plt.plot(short[55535:55535+1000])
rms = np.sqrt(mean_squared_error(org[55500:55500+1000], short[55500:55500+1000]))
plt.title("short error")
plt.xlabel("t")
plt.ylabel("error")
plt.grid()
plt.show()
print("rms for short: ", rms)
wave.write("filled_short2.wav", 8000, short)


plt.plot(org[55500:55500+1000]-long[55500:55500+1000])
# plt.plot(short[55535:55535+1000])
plt.title("long error")
plt.xlabel("t")
plt.ylabel("error")
plt.grid()
plt.show()
rms = np.sqrt(mean_squared_error(org[55500:55500+1000], long[55500:55500+1000]))
print("rms for lng: ", rms)
wave.write("filled_long2.wav", 8000, short)