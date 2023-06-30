#coding=utf-8
import numpy as np
import bisect


class Spline:
    """
    Cubic spline class
    """

    def __init__(self, x, y):
        self.a, self.b, self.c, self.d = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.m = np.linalg.solve(A, B)
        self.c = self.m / 2.0

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        """
        Calculate location
        When t exceeds the boundary, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def __search_index(self, x):
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        Calculate the matrix expression A to the left of the equal sign in the second step of the algorithm
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def __calc_B(self, h):
        """
        Calculate the matrix expression B to the right of the equal sign in the second step of the algorithm
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 6.0 * (self.a[i + 2] - self.a[i + 1]) / h[i + 1] - 6.0 * (self.a[i + 1] - self.a[i]) / h[i]
        return B



