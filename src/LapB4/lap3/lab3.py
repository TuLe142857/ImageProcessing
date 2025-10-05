import matplotlib.pyplot as plt
import scipy.io as sio

data = sio.loadmat('coneFundamentals.mat')
print(type(data))
for k, v in data.items():
    print(f'{k}: {v}')

data = data['coneFundamentals']
print(len(data))

wave_lengths = data[:, 0]

l = data[:, 1]
m = data[:, 2]
s = data[:, 3]

plt.figure("lab3", (10, 5))
plt.plot(wave_lengths, l, 'r-', label='l')
plt.plot(wave_lengths, m, 'g-', label='m')
plt.plot(wave_lengths, s, 'b-', label='s')

plt.legend()
plt.xlabel('Wave length')
plt.ylabel('Sensitivity')
plt.grid(True)

plt.show()