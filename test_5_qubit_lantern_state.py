import numpy as np
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

# 5 qubit identity matrix
i_full = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), i_matrix), i_matrix)

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

cz_123 = i_full - 2 * ((i_full - z_1) / 2).dot((i_full - z_2) / 2).dot(((i_full - z_3) / 2))
cz_134 = i_full - 2 * ((i_full - z_1) / 2).dot((i_full - z_3) / 2).dot((i_full - z_4) / 2)
cz_124 = i_full - 2 * ((i_full - z_1) / 2).dot((i_full - z_2) / 2).dot((i_full - z_4) / 2)
cz_235 = i_full - 2 * ((i_full - z_2) / 2).dot((i_full - z_3) / 2).dot((i_full - z_5) / 2)
cz_345 = i_full - 2 * ((i_full - z_3) / 2).dot((i_full - z_4) / 2).dot((i_full - z_5) / 2)
cz_245 = i_full - 2 * ((i_full - z_2) / 2).dot((i_full - z_4) / 2).dot((i_full - z_5) / 2)

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

# For testing purposes:
E_0_32_on_3 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

E_0_32_on_4 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])


H_2_5 = np.kron(np.kron(np.kron(np.kron(i_matrix, h_matrix), i_matrix), i_matrix), i_matrix)
H_2_4 = np.kron(np.kron(np.kron(i_matrix, h_matrix), i_matrix), i_matrix)
H_2_3 = np.kron(np.kron(i_matrix, h_matrix), i_matrix)
H_3_5 = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), h_matrix), i_matrix), i_matrix)
H_4_5 = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), h_matrix), i_matrix)

# Creation of Hypergraph state
h_state = cz_123.dot(cz_134.dot(cz_124.dot(cz_235.dot(cz_345.dot(cz_245.dot(plus))))))

print(h_state)

p_234 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), i_matrix)
h_state_after_m = p_234.dot(h_state)
h_state_after_m = h_state_after_m / LA.norm(h_state_after_m)
h_state_after_m = np.round(h_state_after_m, 3)
#print(h_state_after_m**2)

# Test 1: 1 qubit Measurement: Compare <h| X_2 |h> and <psi|psi> with |psi> = {extr] H |h>
x_2 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), i_matrix), i_matrix), i_matrix)
prob_exp = np.transpose(h_state).dot(x_2.dot(h_state))
psi = E_0_32.dot(H_2_5.dot(h_state))
prob_extr = np.transpose(psi).dot(psi)
if np.isclose(prob_exp, prob_extr):
    print("Test 1 passed with <h| X_2 |h> = ", prob_extr)
else:
    print("Test 1 failed with expectation probability: ",  prob_exp, " and extraction probability:", prob_extr)

# Test 2: Continued Measurement: Compare <h_post_2 | X_3 | h_post_2> and <psi|psi> with |psi> = {extr] H |h_post_2>
# on state after X_2 measurement
h_state_post_2 = E_0_32.dot(H_2_5.dot(x_2.dot(h_state)))
x_3 = np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), i_matrix), i_matrix)
prob_exp = np.transpose(h_state_post_2).dot(x_3.dot(h_state_post_2))
psi = E_0_16.dot(H_2_4.dot(psi))
prob_extr = np.transpose(psi).dot(psi)
if np.isclose(prob_exp, prob_extr):
    print("Test 2 passed with <h_post_2 | X_3 | h_post_2> = ", prob_extr)
else:
    print("Test 2 failed with expectation probability: ",  prob_exp, " and extraction probability:", prob_extr)

# Test 3: Continued Measurement: Compare <h_post_23 | X_4 |h_post_23 > and <psi|psi> with |psi> = {extr] H |h_post_2>
# on state after X_2 and X_3 measurement
h_state_post_23 = E_0_16.dot(H_2_4.dot(x_3.dot(h_state_post_2)))
x_4 = np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), i_matrix)
prob_exp = np.transpose(h_state_post_23).dot(x_4.dot(h_state_post_23))
psi = E_0_8.dot(H_2_3.dot(psi))
prob_extr = np.transpose(psi).dot(psi)
if np.isclose(prob_exp, prob_extr):
    print("Test 3 passed with <h_post_23 | X_4 |h_post_23 > = ", prob_extr)
else:
    print("Test 3 failed with expectation probability: ",  prob_exp, " and extraction probability:", prob_extr)

# Test 4: Multiple Qubits: Compare <h| X_23 |h> and <psi|psi> with |psi> = {extr] H |h>
# where X_23 is a measurement on X_2 and X_3
x_23 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), i_matrix), i_matrix)
prob_exp = np.transpose(h_state).dot(x_23.dot(h_state))
psi = E_0_16.dot(H_2_4.dot(E_0_32.dot(H_2_5.dot(h_state))))
prob_extr = np.transpose(psi).dot(psi)
if np.isclose(prob_exp, prob_extr):
    print("Test 4 passed with <h| X_23 |h> = ", prob_extr)
else:
    print("Test 4 failed with expectation probability: ",  prob_exp, " and extraction probability:", prob_extr)

# Test 5: Multiple Qubits: Compare <h| X_234 |h> and <psi|psi> with |psi> = {extr] H |h>
# where X_234 is a measurement on X_2, X_3 and X_4
x_234 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), i_matrix)
prob_exp = np.transpose(h_state).dot(x_234.dot(h_state))
psi = E_0_8.dot(H_2_3.dot(E_0_16.dot(H_2_4.dot(E_0_32.dot(H_2_5.dot(h_state))))))
prob_extr = np.transpose(psi).dot(psi)
if np.isclose(prob_exp, prob_extr):
    print("Test 5 passed with <h| X_234 |h> = ", prob_extr)
else:
    print("Test 5 failed with expectation probability: ",  prob_exp, " and extraction probability:", prob_extr)

# Test 6: 1 qubit Measurement: Compare <h| X_3 |h> and <psi|psi> with |psi> = {extr] H |psi>
x_3 = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), (i_matrix+x_matrix)/2), i_matrix), i_matrix)
prob_exp = np.transpose(h_state).dot(x_3.dot(h_state))
psi = E_0_32_on_3.dot(H_3_5.dot(h_state))
prob_extr = np.transpose(psi).dot(psi)
if np.isclose(prob_exp, prob_extr):
    print("Test 6 passed with <h| X_3 |h> = ", prob_extr)
else:
    print("Test 6 failed with expectation probability: ",  prob_exp, " and extraction probability:", prob_extr)

# Test 7: 1 qubit Measurement: Compare <h| X_4 |h> and <psi|psi> with |psi> = {extr] H |psi>
x_4 = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), (i_matrix+x_matrix)/2), i_matrix)
prob_exp = np.transpose(h_state).dot(x_4.dot(h_state))
psi = E_0_32_on_4.dot(H_4_5.dot(h_state))
prob_extr = np.transpose(psi).dot(psi)
if np.isclose(prob_exp, prob_extr):
    print("Test 7 passed with <h| X_4 |h> = ", prob_extr)
else:
    print("Test 7 failed with expectation probability: ",  prob_exp, " and extraction probability:", prob_extr)

# Test 8: 1 Qubit, different basis: Compare <h| Z_2 |h> and <psi|psi> with |psi> = {extr] H |psi>
z_2 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix + z_matrix) / 2), i_matrix), i_matrix), i_matrix)
prob_exp = np.transpose(h_state).dot(z_2.dot(h_state))
psi = E_0_32.dot(h_state)
prob_extr = np.transpose(psi).dot(psi)
if np.isclose(prob_exp, prob_extr):
    print("Test 8 passed with <h| Z_2 |h> = ", prob_extr)
else:
    print("Test 8 failed with expectation probability: ", prob_exp, " and extraction probability:", prob_extr)

# Test 9: Multiple Qubits, different basis: Compare <h| Z_23 |h> and <psi|psi> with |psi> = {extr] H |h>
# where Z_23 is a measurement on Z_2 and Z_3
z_23 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+z_matrix)/2), (i_matrix+z_matrix)/2), i_matrix), i_matrix)
prob_exp = np.transpose(h_state).dot(z_23.dot(h_state))
psi = E_0_16.dot(E_0_32.dot(h_state))
prob_extr = np.transpose(psi).dot(psi)
if np.isclose(prob_exp, prob_extr):
    print("Test 9 passed with <h| Z_23 |h> = ", prob_extr)
else:
    print("Test 9 failed with expectation probability: ",  prob_exp, " and extraction probability:", prob_extr)

# Test 10: Multiple Qubits, different basis: Compare <h| Z_234 |h> and <psi|psi> with |psi> = {extr] H |h>
# where Z_234 is a measurement on Z_2, Z_3 and Z_4
z_234 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+z_matrix)/2), (i_matrix+z_matrix)/2), (i_matrix+z_matrix)/2), i_matrix)
prob_exp = np.transpose(h_state).dot(z_234.dot(h_state))
psi = E_0_8.dot(E_0_16.dot(E_0_32.dot(h_state)))
prob_extr = np.transpose(psi).dot(psi)
if np.isclose(prob_exp, prob_extr):
    print("Test 10 passed with <h| Z_234 |h> = ", prob_extr)
else:
    print("Test 10 failed with expectation probability: ",  prob_exp, " and extraction probability:", prob_extr)

# Test 11: 1 qubit Measurement: Compare <h| X_2' |h> and <psi|psi> with |psi> = {extr] H |h>
x_2 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix-x_matrix)/2), i_matrix), i_matrix), i_matrix)
prob_exp = np.transpose(h_state).dot(x_2.dot(h_state))
psi = E_1_32.dot(H_2_5.dot(h_state))
prob_extr = np.transpose(psi).dot(psi)
if np.isclose(prob_exp, prob_extr):
    print("Test 11 passed with <h| X_2' |h> = ", prob_extr)
else:
    print("Test 11 failed with expectation probability: ",  prob_exp, " and extraction probability:", prob_extr)

# Test 12: Continued Measurement: Compare <h_post_2 | X_3 | h_post_2> and <psi|psi> with |psi> = {extr] H |h_post_2>
# on state after X_2 measurement
h_state_post_2_prime = E_1_32.dot(H_2_5.dot(x_2.dot(h_state)))
x_3 = np.kron(np.kron(np.kron(i_matrix, (i_matrix-x_matrix)/2), i_matrix), i_matrix)
prob_exp = np.transpose(h_state_post_2_prime).dot(x_3.dot(h_state_post_2_prime))
psi = E_1_16.dot(H_2_4.dot(psi))
prob_extr = np.transpose(psi).dot(psi)
if np.isclose(prob_exp, prob_extr):
    print("Test 12 passed with <h_post_2 | X_3' | h_post_2> = ", prob_extr)
else:
    print("Test 12 failed with expectation probability: ",  prob_exp, " and extraction probability:", prob_extr)

# Test 13: Continued Measurement: Compare <h_post_23 | X_4 |h_post_23 > and <psi|psi> with |psi> = {extr] H |h_post_2>
# on state after X_2 and X_3 measurement
h_state_post_23_prime = E_1_16.dot(H_2_4.dot(x_3.dot(h_state_post_2_prime)))
x_4 = np.kron(np.kron(i_matrix, (i_matrix-x_matrix)/2), i_matrix)
prob_exp = np.transpose(h_state_post_23_prime).dot(x_4.dot(h_state_post_23_prime))
psi = E_1_8.dot(H_2_3.dot(psi))
prob_extr = np.transpose(psi).dot(psi)
if np.isclose(prob_exp, prob_extr):
    print("Test 13 passed with <h_post_23 | X_4' |h_post_23 > = ", prob_extr)
else:
    print("Test 13 failed with expectation probability: ",  prob_exp, " and extraction probability:", prob_extr)

# Test 14: Multiple Qubits: Compare <h| X_234' |h> and <psi|psi> with |psi> = {extr] H |h>
# where X_234 is a measurement on X_2, X_3 and X_4
x_23 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix-x_matrix)/2), (i_matrix-x_matrix)/2), i_matrix), i_matrix)
prob_exp = np.transpose(h_state).dot(x_23.dot(h_state))
psi = E_1_16.dot(H_2_4.dot(E_1_32.dot(H_2_5.dot(h_state))))
prob_extr = np.transpose(psi).dot(psi)
if np.isclose(prob_exp, prob_extr):
    print("Test 14 passed with <h| X_23' |h> = ", prob_extr)
else:
    print("Test 14 failed with expectation probability: ",  prob_exp, " and extraction probability:", prob_extr)

# Test 15: Multiple Qubits: Compare <h| X_234' |h> and <psi|psi> with |psi> = {extr] H |h>
# where X_234 is a measurement on X_2, X_3 and X_4
x_234_p = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix-x_matrix)/2), (i_matrix-x_matrix)/2), (i_matrix-x_matrix)/2), i_matrix)
prob_exp = np.transpose(h_state).dot(x_234_p.dot(h_state))
psi = E_1_8.dot(H_2_3.dot(E_1_16.dot(H_2_4.dot(E_1_32.dot(H_2_5.dot(h_state))))))
prob_extr = np.transpose(psi).dot(psi)
if np.isclose(prob_exp, prob_extr):
    print("Test 15 passed with <h| X_234' |h> = ", prob_extr)
else:
    print("Test 15 failed with expectation probability: ",  prob_exp, " and extraction probability:", prob_extr)

# Test 16: Multiple Qubits: Compare <h| X_234' |h> and <psi|psi> with |psi> = {extr] H |h>
# where X_234 is a measurement on X_2, X_3 and X_4
x_234 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix-x_matrix)/2), (i_matrix-x_matrix)/2), i_matrix)
prob_exp = np.transpose(h_state).dot(x_234.dot(h_state))
psi = E_1_8.dot(H_2_3.dot(E_1_16.dot(H_2_4.dot(E_0_32.dot(H_2_5.dot(h_state))))))
prob_extr = np.transpose(psi).dot(psi)
if np.isclose(prob_exp, prob_extr):
    print("Test 16 passed with <h| X_234'' |h> = ", prob_extr)
else:
    print("Test 16 failed with expectation probability: ",  prob_exp, " and extraction probability:", prob_extr)

# Test different measurement outcomes:
x_234 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix-x_matrix)/2), (i_matrix-x_matrix)/2), i_matrix)
prob = np.transpose(h_state).dot(x_234.dot(h_state))
print("Measurement of qubits 2,3,4 as +,-,-:", prob)

x_234 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), (i_matrix-x_matrix)/2), i_matrix)
prob = np.transpose(h_state).dot(x_234.dot(h_state))
print("Measurement of qubits 2,3,4 as +,+,-:", prob)

x_234 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix-x_matrix)/2), (i_matrix+x_matrix)/2), i_matrix)
prob = np.transpose(h_state).dot(x_234.dot(h_state))
print("Measurement of qubits 2,3,4 as +,-,+:", prob)

x_234 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), i_matrix)
prob = np.transpose(h_state).dot(x_234.dot(h_state))
print("Measurement of qubits 2,3,4 as +,+,+:", prob)

x_234 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix-x_matrix)/2), (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), i_matrix)
prob = np.transpose(h_state).dot(x_234.dot(h_state))
print("Measurement of qubits 2,3,4 as -,+,+:", prob)

x_234 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix-x_matrix)/2), (i_matrix-x_matrix)/2), (i_matrix+x_matrix)/2), i_matrix)
prob = np.transpose(h_state).dot(x_234.dot(h_state))
print("Measurement of qubits 2,3,4 as -,-,+:", prob)

x_234 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix-x_matrix)/2), (i_matrix+ x_matrix)/2), (i_matrix-x_matrix)/2), i_matrix)
prob = np.transpose(h_state).dot(x_234.dot(h_state))
print("Measurement of qubits 2,3,4 as -,+,-:", prob)

x_234 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix-x_matrix)/2), (i_matrix-x_matrix)/2), (i_matrix-x_matrix)/2), i_matrix)
prob = np.transpose(h_state).dot(x_234.dot(h_state))
print("Measurement of qubits 2,3,4 as -,-,-:", prob)
