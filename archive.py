from hypergraph import *
from numpy import linalg as LA

# Find hypergraph states on a set Ω = I + M + O of qubits where there exists Bell type entanglement between
# regions I and O after the qubits in M have been measured in the local X-basis

# TASK 1: construct set H of possible hypergraph states

elim_2 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

elim_2_1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

soup = False


def common_eig_vectors(p, q):
    first = True
    out = []
    for i in range(p.shape[0]):
        for j in range(q.shape[0]):
            n = p.shape[1]
            if np.array_equal(p[:, i], q[:, j]):
                if first:
                    out = np.reshape(p[:, i], (n, 1))
                    first = False
                else:
                    out = np.append(out, np.reshape(p[:, i], (n, 1)), axis=1)
    return out


h = Hypergraph(((1,2,3), (1,3,4), (1,2,4), (2,3,5), (3,4,5), (2,4,5)), 5)



w0, v0 = LA.eig(h.stabilizers[0])
w1, v1 = LA.eig(h.stabilizers[1])
w2, v2 = LA.eig(h.stabilizers[2])
w3, v3 = LA.eig(h.stabilizers[3])
w4, v4 = LA.eig(h.stabilizers[4])

d1 = common_eig_vectors(v0, v1)
print(d1)
#d2 = common_eig_vectors(v2, v3)
#print(d2)
#d3 = common_eig_vectors(d2, v3)
#print(common_eig_vectors(d3, v4))

#x = np.array([[5], [2]])
#y = np.array([[100], [200]])
#print(np.append(x, y, axis=1))

#x = np.array([[1,2,3],[4,5,6], [7,8,9]])
#print(np.delete(x, 2, 1))




# TASK 2: construct set M of possible qubits to measure

# TASK 3: Measure each M[i] on each H[i] and check if there exists bell entanglement between qubits I and O



# TESTS:

# Test if the stabilizer is correct

'''x_0 = np.kron(np.kron(np.kron(np.kron(x_matrix, i_matrix), i_matrix), i_matrix), i_matrix)
cz_12 = np.kron(np.kron(np.kron(np.kron(i_matrix,i_matrix), i_matrix), i_matrix), i_matrix) - np.kron(np.kron(np.kron(np.kron(i_matrix, ones_matrix), (z_matrix-i_matrix)), i_matrix), i_matrix)
cz_23 = np.kron(np.kron(np.kron(np.kron(i_matrix,i_matrix), i_matrix), i_matrix), i_matrix) - np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), ones_matrix), (z_matrix-i_matrix)), i_matrix)
cz_13 = np.kron(np.kron(np.kron(np.kron(i_matrix,i_matrix), i_matrix), i_matrix), i_matrix) - np.kron(np.kron(np.kron(np.kron(i_matrix, ones_matrix), i_matrix), (z_matrix-i_matrix)), i_matrix)
s0 = x_0.dot(cz_12).dot(cz_23).dot(cz_13)
print(np.array_equal(s0, h.stabilizers[0]))


x_1 = np.kron(np.kron(np.kron(np.kron(i_matrix, x_matrix), i_matrix), i_matrix), i_matrix)
cz_02 = np.kron(np.kron(np.kron(np.kron(i_matrix,i_matrix), i_matrix), i_matrix), i_matrix) - np.kron(np.kron(np.kron(np.kron(ones_matrix, i_matrix), (z_matrix-i_matrix)), i_matrix), i_matrix)
cz_03 = np.kron(np.kron(np.kron(np.kron(i_matrix,i_matrix), i_matrix), i_matrix), i_matrix) - np.kron(np.kron(np.kron(np.kron(ones_matrix, i_matrix), i_matrix), (z_matrix-i_matrix)), i_matrix)
cz_24 = np.kron(np.kron(np.kron(np.kron(i_matrix,i_matrix), i_matrix), i_matrix), i_matrix) - np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), ones_matrix), i_matrix), (z_matrix-i_matrix))
cz_34 = np.kron(np.kron(np.kron(np.kron(i_matrix,i_matrix), i_matrix), i_matrix), i_matrix) - np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), ones_matrix), (z_matrix-i_matrix))
s1 = x_1.dot(cz_02).dot(cz_03).dot(cz_24).dot(cz_34)
print(np.array_equal(s1, h.stabilizers[1]))


x_4 = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), i_matrix), x_matrix)
cz_12 = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), i_matrix), i_matrix) - np.kron(np.kron(np.kron(np.kron(i_matrix, ones_matrix), (z_matrix - i_matrix)), i_matrix), i_matrix)
cz_23 = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), i_matrix), i_matrix) - np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), ones_matrix), (z_matrix - i_matrix)), i_matrix)
cz_13 = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), i_matrix), i_matrix) - np.kron(np.kron(np.kron(np.kron(i_matrix, ones_matrix), i_matrix), (z_matrix - i_matrix)), i_matrix)
s4 = x_4.dot(cz_12).dot(cz_23).dot(cz_13)
print(np.array_equal(s4, h.stabilizers[4]))'''


# test common_eigenvector

a = np.array([[1,0,0],
              [0,1,0],
              [1,0,2]])

b = np.array([[2,4,0],
              [3,1,0],
              [-1,-4,1]])

a_w, a_v = LA.eig(a)
b_w, b_v = LA.eig(b)


#print(common_eig_vectors(a_v, b_v))

#print(a_v[:,0])
array_poop = np.array([0,0,0])
oop = np.reshape(array_poop, (3,1))
array_poopier = np.array([1,1,1])
oopier = np.reshape(array_poopier, (3,1))
#print(np.append(oop, oopier, axis=1))




def get_cz_matrix(i, j, n):
    p1 = i_matrix
    control = min(i,j)
    target = max(i,j)
    if control == 0:
        p2 = ones_matrix
    else:
        p2 = i_matrix

    for m in range(1, n):
        if m == control:
            p1 = np.kron(p1, i_matrix)
            p2 = np.kron(p2, ones_matrix)
        elif m == target:
            p1 = np.kron(p1, i_matrix)
            p2 = np.kron(p2, z_matrix-i_matrix)
        else:
            p1 = np.kron(p1, i_matrix)
            p2 = np.kron(p2, i_matrix)

    return p1 - p2


# Perform explicitly on 5 qubit lantern state:

# utility

# 5 qubit state in the Pauli-X basis
plus = np.kron(np.kron(np.kron(np.kron(plus_state, plus_state), plus_state), plus_state), plus_state)

# 5 qubit Pauli-X matrices with x_1 only acting on the first qubit and so on
x_1 = np.kron(np.kron(np.kron(np.kron(x_matrix, i_matrix), i_matrix), i_matrix), i_matrix)
x_2 = np.kron(np.kron(np.kron(np.kron(i_matrix, x_matrix), i_matrix), i_matrix), i_matrix)

# 5 qubit Pauli-Z matrices with z_1 only acting on the first qubit and so on

z_1 = np.kron(np.kron(np.kron(np.kron(z_matrix, i_matrix), i_matrix), i_matrix), i_matrix)
z_2 = np.kron(np.kron(np.kron(np.kron(i_matrix, z_matrix), i_matrix), i_matrix), i_matrix)
z_3 = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), z_matrix), i_matrix), i_matrix)
z_4 = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), z_matrix), i_matrix)
z_5 = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), i_matrix), z_matrix)

# 5 qubit identity matrix
i_full = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), i_matrix), i_matrix)

# build hypergraph using cz operators
cz_123 = i_full - 2 * ((i_full - z_1) / 2).dot((i_full - z_2) / 2).dot((i_full - z_3) / 2)
cz_134 = i_full - 2 * ((i_full - z_1) / 2).dot((i_full - z_3) / 2).dot((i_full - z_4) / 2)
cz_124 = i_full - 2 * ((i_full - z_1) / 2).dot((i_full - z_2) / 2).dot((i_full - z_4) / 2)
cz_235 = i_full - 2 * ((i_full - z_2) / 2).dot((i_full - z_3) / 2).dot((i_full - z_5) / 2)
cz_345 = i_full - 2 * ((i_full - z_3) / 2).dot((i_full - z_4) / 2).dot((i_full - z_5) / 2)
cz_245 = i_full - 2 * ((i_full - z_2) / 2).dot((i_full - z_4) / 2).dot((i_full - z_5) / 2)

# prepare state
h_state = cz_123.dot(cz_134.dot(cz_124.dot(cz_235.dot(cz_345.dot(cz_245.dot(plus))))))

# perform measurement on qubits 2,3,4
p_234 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), i_matrix)
h_state = p_234.dot(h_state)
h_state = h_state / LA.norm(h_state)

# extract qubits 0, 4

elim_2 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

# hadamard on qubit 1 acting on 5
hadamard_1_5 = np.kron(np.kron(np.kron(np.kron(i_matrix, h_matrix), i_matrix), i_matrix), i_matrix)
h_state = elim_2.dot(hadamard_1_5.dot(h_state))

#print("First:",h_state)


elim_3 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

hadamard_1_4 = np.kron(np.kron(np.kron(i_matrix, h_matrix), i_matrix), i_matrix)
h_state = elim_3.dot(hadamard_1_4.dot(h_state))

elim_4 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0]])

hadamard_1_3 = np.kron(np.kron(i_matrix, h_matrix), i_matrix)
h_state = elim_4.dot(hadamard_1_3.dot(h_state))

h_state = h_state / LA.norm(h_state)

rho = h_state.dot(np.transpose(h_state))
partial_rho = np.trace(rho.reshape([2, 2, 2, 2]), axis1=0, axis2=2)
w, v = LA.eig(partial_rho)
#print(w)



# CODE I AM REFIGURING

# FROM RUN HYPERGRAPH

# create all possible hypergraphs taking each combination of faces
for i in range(0, len(possible_faces)):
    for faces in itertools.combinations(possible_faces, i+1):
        each_v_in_a_face = True
        #each_v_in_a_face = check_each_v_in_a_face(faces)
        if each_v_in_a_face:
            h = Hypergraph(faces, num_v)
            case_list = []
            prob_list = []
            for case in list(itertools.product([0, 1], repeat=len(m))):
                m_set = {m[0]: case[0], }
                for p in range(1, len(m)):
                    m_set[m[p]] = case[p]
                save_h_state = h.h_state
                m_took = h.perform_measurement(m_set)
                prob = np.transpose(save_h_state).dot(m_took.dot(save_h_state))
                h.normalize()
                h.extract_bits(m_set)
                rho = h.h_state.dot(np.transpose(h.h_state))
                partial_rho = np.trace(rho.reshape([2, 2, 2, 2]), axis1=1, axis2=3)
                w, v = LA.eig(partial_rho)
                if np.allclose(w, np.array([0.5, 0.5])):
                    case_list.append(case)
                    prob_list.append(prob)
                h.h_state = save_h_state
            if np.isclose(1, sum(prob_list)):
                print("Faces: ", faces, " with cases", case_list, "with total prob", sum(prob_list))

# FROM HYPERGRAPH

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
        if LA.norm(self.h_state) != 0:
            self.h_state = self.h_state / LA.norm(self.h_state)

    def perform_measurement(self, m_set):
        if 0 in m_set:
            if m_set.get(0) == 0:
                p = (i_matrix + x_matrix) / 2
            else:
                p = (i_matrix - x_matrix) / 2
        else:
            p = i_matrix
        for i in range(1, self.num_v):
            if i in m_set:
                if m_set[i] == 0:
                    p = np.kron(p, (i_matrix + x_matrix) / 2)
                else:
                    p = np.kron(p, (i_matrix - x_matrix) / 2)
            else:
                p = np.kron(p, i_matrix)
        self.h_state = p.dot(self.h_state)
        return p

    def extract_bits(self, m_set):
        num_bits = self.num_v
        itr = 0
        for i in m_set:
            dim = len(self.h_state)
            sub_size = int(dim / (2 ** (i - itr + 1)))
            sub_i = np.identity(sub_size)
            extr_matrix = np.zeros([int(dim / 2), dim])
            if m_set[i] == 0:
                for k in range(2 ** (i - itr)):
                    extr_matrix[k * sub_size:sub_size * (k + 1), 2 * k * sub_size:sub_size * (2 * k + 1)] = sub_i
            else:
                for k in range(2 ** (i - itr)):
                    extr_matrix[k * sub_size:sub_size * (k + 1), (2 * k + 1) * sub_size:sub_size * (2 * k + 2)] = sub_i
            had_on_i = construct_hadamard(num_bits, i - itr)
            self.h_state = extr_matrix.dot(had_on_i.dot(self.h_state))
            num_bits = num_bits - 1
            itr = itr + 1
