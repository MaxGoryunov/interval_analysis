import numpy as np


class Interval:
    def __init__(self, left, right):
        if left > right:
            raise ValueError("Lower bound cannot be greater than upper bound.")
        self.left = left
        self.right = right

    def __repr__(self):
        return f"[{self.left:.4f}, {self.right:.4f}]"

    def mid(self):
        return (self.left + self.right) / 2

    def width(self):
        return self.right - self.left

    def rad(self):
        return (self.right - self.left) / 2

    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self.left + other.left, self.right + other.right)
        return Interval(self.left + other, self.right + other)

    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(self.left - other.right, self.right - other.left)
        return Interval(self.left - other, self.right - other)

    def __mul__(self, other):
        if isinstance(other, Interval):
            products = np.array([
                self.left * other.left,
                self.left * other.right,
                self.right * other.left,
                self.right * other.right
            ])
            return Interval(np.min(products), np.max(products))
        return Interval(self.left * other, self.right * other)

    def __div__(self, other):
        if isinstance(other, Interval):
            divisions = np.array([
                self.left / other.left,
                self.left / other.right,
                self.right / other.left,
                self.right / other.right
            ])
            return Interval(np.min(divisions), np.max(divisions))
        return Interval(self.left / other, self.right / other)

    def __contains__(self, value):
        return self.left <= value <= self.right

    def __and__(self, other):
        new_left = max(self.left, other.left)
        new_right = min(self.right, other.right)
        if new_left > new_right:
            return Interval(0, 0)
        return Interval(new_left, new_right)

    def __or__(self, other):
        return Interval(
            min(self.left, other.left), max(self.right, other.right)
        )