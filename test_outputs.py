import numpy as np
import itertools
from numpy import linalg as LA

plus_state = np.array([[np.sqrt(.5)],
                       [np.sqrt(.5)]])
minus_state = np.array([[np.sqrt(.5)],
                       [-np.sqrt(.5)]])
zero_state = np.array([[1],
                       [0]])
one_state = np.array([[0],
                      [1]])
i_matrix = np.array([[1, 0],
                     [0, 1]])
z_matrix = np.array([[1, 0],
                     [0, -1]])
x_matrix = np.array([[0, 1],
                     [1, 0]])

h_matrix = np.sqrt(.5)*np.array([[1,1],
                                 [1,-1]])

# 5 qubit state in the Pauli-X basis
plus = np.kron(np.kron(np.kron(np.kron(plus_state, plus_state), plus_state), plus_state), plus_state)

# 6 qubit state in the Pauli-X basis
plus6 = np.kron(np.kron(np.kron(np.kron(np.kron(plus_state, plus_state), plus_state), plus_state), plus_state), plus_state)

# 5 qubit identity matrix
i_full = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), i_matrix), i_matrix)

# 6 qubit identity matrix
i6_full = np.kron(np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), i_matrix), i_matrix), i_matrix)

# 5 qubit Pauli-X matrices with x_2 only acting on the second qubit and so on
x_2 = np.kron(np.kron(np.kron(np.kron(i_matrix, x_matrix), i_matrix), i_matrix), i_matrix)
x_3 = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), x_matrix), i_matrix), i_matrix)
x_4 = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), x_matrix), i_matrix)

# 5 qubit Pauli-Z matrices with z_1 only acting on the first qubit and so on
z_1 = np.kron(np.kron(np.kron(np.kron(z_matrix, i_matrix), i_matrix), i_matrix), i_matrix)
z_2 = np.kron(np.kron(np.kron(np.kron(i_matrix, z_matrix), i_matrix), i_matrix), i_matrix)
z_3 = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), z_matrix), i_matrix), i_matrix)
z_4 = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), z_matrix), i_matrix)
z_5 = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), i_matrix), z_matrix)

# 6 qubit Pauli-Z matrices
z6_0 = np.kron(np.kron(np.kron(np.kron(np.kron(z_matrix, i_matrix), i_matrix), i_matrix), i_matrix), i_matrix)
z6_1 = np.kron(np.kron(np.kron(np.kron(np.kron(i_matrix, z_matrix), i_matrix), i_matrix), i_matrix), i_matrix)
z6_2 = np.kron(np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), z_matrix), i_matrix), i_matrix), i_matrix)
z6_3 = np.kron(np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), z_matrix), i_matrix), i_matrix)
z6_4 = np.kron(np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), i_matrix), z_matrix), i_matrix)
z6_5 = np.kron(np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), i_matrix), i_matrix), z_matrix)

def normalize(state):
    if 0 == np.count_nonzero(state):
        return state
    else:
        return state / LA.norm(state)



i_16 = np.identity(16)
E_0_64 = np.zeros([32, 64])
E_0_64[0:16,0:16] = i_16
E_0_64[16:32,32:48] = i_16



E_0_32 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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

E_1_32 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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

E_0_16 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

E_1_16 = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

E_0_8 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0]])

E_1_8 = np.array([[0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1]])

H_2_6 = np.kron(np.kron(np.kron(np.kron(np.kron(i_matrix, h_matrix), i_matrix), i_matrix), i_matrix), i_matrix)
H_2_5 = np.kron(np.kron(np.kron(np.kron(i_matrix, h_matrix), i_matrix), i_matrix), i_matrix)
H_2_4 = np.kron(np.kron(np.kron(i_matrix, h_matrix), i_matrix), i_matrix)
H_2_3 = np.kron(np.kron(i_matrix, h_matrix), i_matrix)

cz_014 = i_full - 2 * ((i_full - z_1) / 2).dot((i_full - z_2) / 2).dot(((i_full - z_5) / 2))
cz_123 = i_full - 2 * ((i_full - z_2) / 2).dot((i_full - z_3) / 2).dot(((i_full - z_4) / 2))
cz_024 = i_full - 2 * ((i_full - z_1) / 2).dot((i_full - z_3) / 2).dot(((i_full - z_5) / 2))
cz_012 = i_full - 2 * ((i_full - z_1) / 2).dot((i_full - z_2) / 2).dot(((i_full - z_3) / 2))
cz_234 = i_full - 2 * ((i_full - z_3) / 2).dot((i_full - z_4) / 2).dot(((i_full - z_5) / 2))

# CHECK 6 QUBIT OUTPUT

#Faces:  ((0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 1, 5), (0, 2, 3), (0, 2, 4), (0, 2, 5), (1, 2, 5), (1, 3, 4), (2, 3, 4))  with cases [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 1, 1), (1, 0, 0, 0), (1, 0, 1, 1), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)] with total prob [[1.]]

cz6_012 = i6_full - 2 * ((i6_full - z6_0) / 2).dot((i6_full - z6_1) / 2).dot(((i6_full - z6_2) / 2))
cz6_013 = i6_full - 2 * ((i6_full - z6_0) / 2).dot((i6_full - z6_1) / 2).dot(((i6_full - z6_3) / 2))
cz6_014 = i6_full - 2 * ((i6_full - z6_0) / 2).dot((i6_full - z6_1) / 2).dot(((i6_full - z6_4) / 2))
cz6_015 = i6_full - 2 * ((i6_full - z6_0) / 2).dot((i6_full - z6_1) / 2).dot(((i6_full - z6_5) / 2))
cz6_023 = i6_full - 2 * ((i6_full - z6_0) / 2).dot((i6_full - z6_2) / 2).dot(((i6_full - z6_3) / 2))
cz6_024 = i6_full - 2 * ((i6_full - z6_0) / 2).dot((i6_full - z6_2) / 2).dot(((i6_full - z6_4) / 2))
cz6_025 = i6_full - 2 * ((i6_full - z6_0) / 2).dot((i6_full - z6_2) / 2).dot(((i6_full - z6_5) / 2))
cz6_125 = i6_full - 2 * ((i6_full - z6_1) / 2).dot((i6_full - z6_2) / 2).dot(((i6_full - z6_5) / 2))
cz6_134 = i6_full - 2 * ((i6_full - z6_1) / 2).dot((i6_full - z6_3) / 2).dot(((i6_full - z6_4) / 2))
cz6_234 = i6_full - 2 * ((i6_full - z6_2) / 2).dot((i6_full - z6_3) / 2).dot(((i6_full - z6_4) / 2))

h_state =  cz6_012.dot(cz6_013.dot(cz6_014.dot(cz6_015.dot(cz6_023.dot(cz6_024.dot(cz6_025.dot(cz6_125.dot(cz6_134.dot(cz6_234.dot(plus6))))))))))
p_1234 = np.kron(np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2),(i_matrix-x_matrix)/2), i_matrix)
print(h_state)
h_state = p_1234.dot(h_state)
print(h_state)
h_state = h_state / LA.norm(h_state)
case = E_1_8.dot(H_2_3.dot(E_0_16.dot(H_2_4.dot(E_0_32.dot(H_2_5.dot(E_0_64.dot(H_2_6.dot(h_state))))))))
rho = case.dot(np.transpose(case))
partial_rho = np.array([[rho[0][0]+rho[1][1], rho[0][2]+rho[1][3]],
                        [rho[2][0]+rho[3][1], rho[2][2]+rho[3][3]]])
w, v = LA.eig(partial_rho)
print(w)
if np.allclose(w, np.array([0.5,0.5])):
    print("Case 0 success")

'''
# CHECK 5 QUBIT OUTPUTS

# faces ((0, 1, 4), (1, 2, 3)) with cases [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
h_state = cz_014.dot(cz_123.dot(plus))
p_123 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix-x_matrix)/2), (i_matrix+x_matrix)/2), i_matrix)
h_state = p_123.dot(h_state)
h_state = h_state / LA.norm(h_state)
case = E_0_8.dot(H_2_3.dot(E_1_16.dot(H_2_4.dot(E_0_32.dot(H_2_5.dot(h_state))))))
rho = case.dot(np.transpose(case))
partial_rho = np.array([[rho[0][0]+rho[1][1], rho[0][2]+rho[1][3]],
                        [rho[2][0]+rho[3][1], rho[2][2]+rho[3][3]]])
w, v = LA.eig(partial_rho)
if np.allclose(w, np.array([0.5,0.5])):
    print("Case 0 success")

# faces ((0, 2, 4), (1, 2, 3)) with cases [(0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
h_state = cz_123.dot(cz_024.dot(plus))
p_123 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix-x_matrix)/2), (i_matrix-x_matrix)/2), (i_matrix-x_matrix)/2), i_matrix)
h_state = p_123.dot(h_state)
h_state = h_state / LA.norm(h_state)
case = E_1_8.dot(H_2_3.dot(E_1_16.dot(H_2_4.dot(E_1_32.dot(H_2_5.dot(h_state))))))
rho = case.dot(np.transpose(case))
partial_rho = np.array([[rho[0][0]+rho[1][1], rho[0][2]+rho[1][3]],
                        [rho[2][0]+rho[3][1], rho[2][2]+rho[3][3]]])
w, v = LA.eig(partial_rho)
if np.allclose(w, np.array([0.5,0.5])):
    print("Case 1 success")

# faces ((0, 1, 2), (0, 1, 4), (2, 3, 4)) with cases [(0, 1, 0)]
h_state = cz_012.dot(cz_014.dot(cz_234.dot(plus)))
p_123 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix-x_matrix)/2), (i_matrix+x_matrix)/2), i_matrix)
h_state = p_123.dot(h_state)
h_state = h_state / LA.norm(h_state)
case = E_0_8.dot(H_2_3.dot(E_1_16.dot(H_2_4.dot(E_0_32.dot(H_2_5.dot(h_state))))))
rho = case.dot(np.transpose(case))
partial_rho = np.array([[rho[0][0]+rho[1][1], rho[0][2]+rho[1][3]],
                        [rho[2][0]+rho[3][1], rho[2][2]+rho[3][3]]])
w, v = LA.eig(partial_rho)
if np.allclose(w, np.array([0.5,0.5])):
    print("Case 2 success")'''