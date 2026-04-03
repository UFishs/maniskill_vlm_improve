import pandas as pd
import matplotlib.pyplot as plt

# x_step = [200, 429, 500, 840]
# x_label = ['200', '429', '500', '830']

# y_motion = [0.225, 0.82, 0.89, None]
# y_system = [0.225, 0.84, None, 0.93]

# x_step = [200, 492, 500]
# x_label = ['200', '492', '500']

# y_motion = [0.705, None, 0.84]
# y_system = [0.705, 0.8, None]
# y_primitive = [0.645, 0.72, None]

x_step = [200, 429, 840]
x_label = ['iter_0', 'iter_1', 'iter_2']

y_motion = [0.225, 0.86, 0.95]
y_system = [0.225, 0.82, 0.93]
y_boot = [0.225, 0.54, 0.6]



# x_step = [200, 492, 893]
# x_label = ['iter_0', 'iter_1', 'iter_2']

# y_motion = [0.705, 0.84, 0.895]
# y_system = [0.705, 0.8, 0.88]
# y_boot = [0.705, 0.76, 0.765]


# x_step = [200, 484, 861]
# x_label = ['iter_0', 'iter_1', 'iter_2']

# y_motion = [0.13, 0.625, 0.73]
# y_system = [0.13, 0.61, 0.735]
# y_boot = [0.13, 0.23, 0.25]


def remove_none(x, y):
    xs, ys = [], []
    for xi, yi in zip(x, y):
        if yi is not None:
            xs.append(xi)
            ys.append(yi)
    return xs, ys

x_m, y_m = remove_none(x_step, y_motion)
x_s, y_s = remove_none(x_step, y_system)
# x_p, y_p = remove_none(x_step, y_primitive)
x_b, y_b = remove_none(x_step, y_boot)

plt.plot(x_m, y_m, marker='o', label='Expert')
plt.plot(x_s, y_s, marker='o', label='Ours')
plt.plot(x_b, y_b, marker='o', label='Bootstrap')
# plt.plot(x_p, y_p, marker='o', label='primitive')


plt.xticks(x_step, x_label)
plt.xlabel('#iter')
plt.ylabel('success rate')
plt.legend()

plt.savefig('draw.png')
