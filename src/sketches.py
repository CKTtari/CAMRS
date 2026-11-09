import math
import mmh3
import numpy as np


class CountMinSketch:
    def __init__(self, width, depth, seed=42):
        self.width = width
        self.depth = depth
        self.seed = seed
        self.table = np.zeros((depth, width), dtype=np.int64)
        self.total = 0

    def _hash(self, item, d):
        return mmh3.hash(str(item).encode(), d + self.seed) % self.width

    def add(self, item, delta=1):
        for d in range(self.depth):
            idx = self._hash(item, d)
            self.table[d, idx] += delta
        self.total += delta

    def estimate(self, item):
        vals = np.array([self.table[d, self._hash(item, d)] for d in range(self.depth)])
        return int(vals.min())

    def merge(self, other):
        assert self.width == other.width and self.depth == other.depth
        self.table += other.table
        self.total += other.total

    def estimate_in_bucket(self, bucket_id):
        return int(self.table[:, bucket_id].min())

    def __sizeof__(self):
        return self.table.nbytes


class HyperLogLog:
    def __init__(self, b=12):
        self.b = b
        self.m = 1 << b
        self.registers = np.zeros(self.m, dtype=np.uint8)
        self.alpha = self._get_alpha()

    def _get_alpha(self):
        if self.m == 16:
            return 0.673
        elif self.m == 32:
            return 0.697
        elif self.m == 64:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / self.m)

    def add(self, item):
        x = mmh3.hash(str(item).encode())
        idx = x >> (32 - self.b)
        w = x & ((1 << (32 - self.b)) - 1)
        leading_zeros = 1
        while leading_zeros < (32 - self.b) and (w >> (31 - self.b - leading_zeros)) & 1 == 0:
            leading_zeros += 1
        self.registers[idx] = max(self.registers[idx], leading_zeros)

    def merge(self, other):
        assert len(self.registers) == len(other.registers)
        self.registers = np.maximum(self.registers, other.registers)

    def estimate(self):
        Z = 1.0 / np.sum(2.0 ** (-self.registers.astype(np.float64)))
        E = self.alpha * self.m * self.m * Z
        if E <= 2.5 * self.m:
            V = np.sum(self.registers == 0)
            if V > 0:
                E = self.m * math.log(self.m / V)
        return int(E)

    def __sizeof__(self):
        return self.registers.nbytes
