import cubic_spline
import numpy as np
from matplotlib import pyplot as plt

def main():
    x = [-4., -2, 0.0, 2, 4, 6, 10]
    y = [1.2, 0.6, 0.0, 1.5, 3.8, 5.0, 3.0]

    spline = cubic_spline.Spline(x, y)
    rx = np.arange(-4.0, 10, 0.01)
    ry = [spline.calc(i) for i in rx]

    plt.plot(x, y, "og")
    plt.plot(rx, ry, "-r")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    main()