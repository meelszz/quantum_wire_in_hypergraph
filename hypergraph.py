import numpy as np
from numpy import linalg as LA
import itertools


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


def construct_hadamard(num_v, q):
    if q == 0:
        out = h_matrix
    else:
        out = i_matrix
    for i in range(1, num_v):
        if i == q:
            out = np.kron(out, h_matrix)
        else:
            out = np.kron(out, i_matrix)
    return out


# Create n qubit identity matrix : O(n)
def initialize_i(num_v):
    i = i_matrix
    for n in range(num_v - 1):
        i = np.kron(i, i_matrix)
    return i


# Initialize state as |+>_n : O(n)
def initialize_state(num_v):
    state = plus_state
    for n in range(num_v-1):
        state = np.kron(state, plus_state)
    return state


# Create matrices Z_1 to Z_n : O(n^2)
def initialize_loz(num_v):
    loz = [0] * num_v
    loz[0] = z_matrix
    for k in range(1, num_v):
        loz[0] = np.kron(loz[0], i_matrix)
    for n in range(1, num_v):
        loz[n] = i_matrix
        for k in range(1, num_v):
            if n == k:
                loz[n] = np.kron(loz[n], z_matrix)
            else:
                loz[n] = np.kron(loz[n], i_matrix)
    return loz


# Create measurement operators
def create_m_ops(num_v):
    set_p = {}
    for case in list(itertools.product([0, 1], repeat=int(num_v - 2))):
        # create unique key for case
        key = gen_case_key(case)
        # create proj matrix for measurement case
        p = i_matrix
        for c in case:
            if c == 1:
                p = np.kron(p, (i_matrix - x_matrix) / 2)
            else:
                p = np.kron(p, (i_matrix + x_matrix) / 2)
        p = np.kron(p, i_matrix)
        # update proj matrix set for case
        set_p[key] = p
    return set_p


# Create extraction matrices
# extr_ms[(n-1)th] is the nth extraction matrix assuming measurement result 0
# extr_ms[2*(n-1)th] is the nth extraction matrix assuming measurement result 1
def create_extr_matrices(num_v):
    extr_ms = [0] * (num_v - 2) * 2
    for a in range(num_v - 2):
        id = np.identity((int(2 ** (num_v - a) / 4)))
        dim = int(2 ** (num_v - a))
        dim2 = int((1 / 2) * (2 ** (num_v - a)))
        dim3 = int((3 / 4) * (2 ** (num_v - a)))
        dim4 = int((1 / 4) * (2 ** (num_v - a)))
        extr_ms[a] = np.zeros([dim2, dim])
        extr_ms[a][0:dim4, 0:dim4] = id
        extr_ms[a][dim4:dim2, dim2:dim3] = id
        index = (num_v - 2 + a)
        extr_ms[index] = np.zeros([dim2, dim])
        extr_ms[index][0:dim4, dim4:dim2] = id
        extr_ms[index][dim4:dim2, dim3:dim] = id
    return extr_ms


# Create Hadamard matrices used with extraction matrices - on qubit 1 with decreasing # of qubits
def create_loh(num_v):
    loh = [0] * (num_v - 2)
    for i in range(num_v-2):
        loh[i] = construct_hadamard(num_v-i, 1)
    return loh


# Generate unique key based on case
def gen_case_key(case):
    primes = (5, 7, 11, 13, 17)
    key = 0
    for i in range(len(case)):
        key = key + ((case[i]+1) * primes[i])
    return key


class Hypergraph:
    def __init__(self, num_v):
        self.num_v = num_v
        self.loz = initialize_loz(num_v)
        self.i = initialize_i(num_v)
        self.m_ops = create_m_ops(num_v)
        self.extr_ms = create_extr_matrices(num_v)
        self.loh = create_loh(num_v)
        self.h_state = 0

    # Construct state from CZ matrices : O(n) for small cases
    def construct_state(self, faces):
        self.h_state = initialize_state(self.num_v)
        for f in faces:
            cz = self.i - 2 * ((self.i - self.loz[f[0]]) / 2).dot((self.i - self.loz[f[1]]) / 2).dot(((self.i - self.loz[f[2]]) / 2))
            self.h_state = cz.dot(self.h_state)

    def normalize(self):
        if LA.norm(self.h_state) != 0:
            self.h_state = self.h_state / LA.norm(self.h_state)

    # Perform measurement on state : O(n)
    def perform_measurement(self, case):
        key = gen_case_key(case)
        p = self.m_ops[key]
        self.h_state = p.dot(self.h_state)
        self.normalize()
        return p

    # Extract bits : O(n)
    def extract_bits(self, case):
        for i in range(self.num_v - 2):
            had = self.loh[i]
            if case[i] == 0:
                extr_m_0 = self.extr_ms[i]
                self.h_state = extr_m_0.dot(had.dot(self.h_state))
            else:
                index = self.num_v - 2 + i
                extr_m_1 = self.extr_ms[index]
                self.h_state = extr_m_1.dot(had.dot(self.h_state))