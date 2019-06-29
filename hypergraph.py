import numpy as np
from numpy import linalg as LA
import math

#  A hypergraph consists of its faces (h_graph) the number of qubits
#  it has (num_v) and the generators of it's stabilizer

# utility
x_matrix = np.array([[0, 1],
                     [1, 0]])

i_matrix = np.array([[1, 0],
                     [0, 1]])

ones_matrix = np.array([[0, 0],
                        [0, 1]])

z_matrix = np.array([[1, 0],
                     [0, -1]])

h_matrix = (np.sqrt(.5))*np.array([[1,1],
                                       [1,-1]])

plus_state = np.array([[np.sqrt(.5)],
                       [np.sqrt(.5)]])


def construct_hadamard(dim, q):
    if q == 0:
        out = h_matrix
    else:
        out = i_matrix
    for i in range(1, dim):
        if i == q:
            out = np.kron(out, h_matrix)
        else:
            out = np.kron(out, i_matrix)
    return out


class Hypergraph:
    def __init__(self, faces, num_v):
        self.faces = faces
        self.num_v = num_v
        self.loz = [0] * self.num_v
        self.i = i_matrix
        self.initialize_i()
        self.h_state = plus_state
        self.create_state()

    def initialize_i(self):
        for n in range(self.num_v -1):
            self.i = np.kron(self.i, i_matrix)

    def create_state(self):
        for n in range(self.num_v):
            if n != 0:
                self.h_state = np.kron(self.h_state, plus_state)
            if n == 0:
                self.loz[0] = z_matrix
            else:
                self.loz[n] = i_matrix
            for k in range(1, self.num_v):
                if n == k:
                    self.loz[n] = np.kron(self.loz[n], z_matrix)
                else:
                    self.loz[n] = np.kron(self.loz[n], i_matrix)
        for f in self.faces:
            cz = self.i - 2 * ((self.i - self.loz[f[0]]) / 2).dot((self.i - self.loz[f[1]]) / 2).dot(((self.i - self.loz[f[2]]) / 2))
            self.h_state = cz.dot(self.h_state)

    def normalize(self):
        for i in range(np.shape(self.h_state)[0]):
            if LA.norm(self.h_state[i]) != 0:
                self.h_state[i] = self.h_state[i] / LA.norm(self.h_state[i])

    def perform_measurement(self, m_set):
        if 0 in m_set:
            p = (x_matrix + i_matrix)/2
        else:
            p = i_matrix
        for i in range(1, self.num_v):
            if i in m_set:
                p = np.kron(p, (x_matrix + i_matrix)/2)
            else:
                p = np.kron(p, i_matrix)
        self.h_state = p.dot(self.h_state)

    def extract_bits(self, m_set):
        los = []
        self.h_state = [self.h_state]
        for i in range(len(m_set)):
            self.extract(m_set[i] - i, los)

    def extract(self, i, los):
        dim = len(self.h_state[0])
        sub_size = int(dim/(2**(i+1)))
        sub_i = np.identity(sub_size)
        extr_matrix_0 = np.zeros([int(dim/2), dim])
        extr_matrix_1 = np.zeros([int(dim/ 2), dim])
        for k in range(2**i):
            extr_matrix_0[k * sub_size:sub_size * (k + 1), 2 * k * sub_size:sub_size*(2*k + 1)] = sub_i
            extr_matrix_1[k * sub_size:sub_size * (k + 1), (2 * k + 1) * sub_size:sub_size * (2 * k + 2)] = sub_i
        had_on_i = construct_hadamard(self.num_v, i)
        num_s = np.shape(self.h_state)[0]
        for j in range(num_s):
            self.h_state.append(extr_matrix_1.dot(had_on_i.dot(self.h_state[j])))
            self.h_state[j] = extr_matrix_0.dot(had_on_i.dot(self.h_state[j]))
        self.num_v = self.num_v - 1
