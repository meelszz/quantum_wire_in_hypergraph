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

h_state = cz_123.dot(cz_134.dot(cz_124.dot(cz_235.dot(cz_345.dot(cz_245.dot(plus))))))


def normalize(state):
    if 0 == np.count_nonzero(state):
        return state
    else:
        return state / LA.norm(state)

# Check <h|X_2|h> = <h|X_3|h> = <h|X_4|h> = 1
x_2_check = np.transpose(h_state).dot(x_2.dot(h_state))
#print("x2 check: ", x_2_check)
x_3_check = np.transpose(h_state).dot(x_2.dot(h_state))
#print("x3 check: ", x_3_check)
x_4_check = np.transpose(h_state).dot(x_2.dot(h_state))
#print("x4 check: ", x_4_check)

outcome_00000 = np.kron(np.kron(np.kron(np.kron(zero_state, zero_state), zero_state), zero_state), zero_state)

outcome_plus = np.kron(np.kron(np.kron(np.kron(plus_state, plus_state), plus_state), plus_state), plus_state)
h_2_5 = np.kron(np.kron(np.kron(np.kron(i_matrix, h_matrix), i_matrix), i_matrix), i_matrix)
h_3_5 = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), h_matrix), i_matrix), i_matrix)
h_4_5 = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), h_matrix), i_matrix)
h = h_2_5.dot(h_3_5.dot(h_4_5))

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

H_2_5 = np.kron(np.kron(np.kron(np.kron(i_matrix, h_matrix), i_matrix), i_matrix), i_matrix)
H_2_4 = np.kron(np.kron(np.kron(i_matrix, h_matrix), i_matrix), i_matrix)
H_2_3 = np.kron(np.kron(i_matrix, h_matrix), i_matrix)

'''
# Test case 2: 2nd qubit measured 0, 3rd qubit measured 0 and 4th qubit measured 1
h_state_temp = h_state
print("TEST")
x_2 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), i_matrix), i_matrix), i_matrix)
prob_x2 = np.transpose(h_state_temp).dot(x_2.dot(h_state_temp))
print(prob_x2)
h_state_temp = E_0_32.dot(H_2_5.dot(x_2.dot(h_state_temp)))
x_3 = np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), i_matrix), i_matrix)
prob_x3 = np.transpose(h_state_temp).dot(x_3.dot(h_state_temp))
print(prob_x3)
h_state_temp = E_0_16.dot(H_2_4.dot(x_3.dot(h_state_temp)))
x_4 = np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), i_matrix)
prob_x4 = np.transpose(h_state_temp).dot(x_4.dot(h_state_temp))
print(prob_x4)
h_state_temp = E_1_8.dot(H_2_3.dot(x_4.dot(h_state_temp)))
#print(h_state_temp)

#print(E_1_8.dot(H_2_3.dot(x_4.dot(E_0_16.dot(H_2_4.dot(x_3.dot(E_0_32.dot(H_2_5.dot(x_2.dot(h_state))))))))))

p_123 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), (i_matrix-x_matrix)/2), i_matrix)
h_state = p_123.dot(h_state)
h_state = h_state / LA.norm(h_state)
print(E_1_8.dot(H_2_3.dot(E_0_16.dot(H_2_4.dot(E_0_32.dot(H_2_5.dot(h_state)))))))
'''

# Perform measurement and normalize

# Extract qubits 1 and 5 under diff measurment outcomes

# Case 1: 2nd qubit measured 0, 3rd qubit measured 0 and 4th qubit measured 0
p_123 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), i_matrix)
h_state1 = p_123.dot(h_state)
h_state1 = h_state1 / LA.norm(h_state1)

case = E_0_8.dot(H_2_3.dot(E_0_16.dot(H_2_4.dot(E_0_32.dot(H_2_5.dot(h_state1))))))
rho = case.dot(np.transpose(case))
partial_rho = np.array([[rho[0][0]+rho[1][1], rho[0][2]+rho[1][3]],
                        [rho[2][0]+rho[3][1], rho[2][2]+rho[3][3]]])
w, v = LA.eig(partial_rho)
if np.allclose(w, np.array([0.5,0.5])):
    print("Case 1 success")
else:
    print("Case 1 fail")
print(case)
print(w)

# Case 2: 2nd qubit measured 0, 3rd qubit measured 0 and 4th qubit measured 1
p_123 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), (i_matrix-x_matrix)/2), i_matrix)
h_state2 = p_123.dot(h_state)
h_state2 = h_state2 / LA.norm(h_state2)
case = E_1_8.dot(H_2_3.dot(E_0_16.dot(H_2_4.dot(E_0_32.dot(H_2_5.dot(h_state2))))))
print("Hadamard matrix ", 0)
print(H_2_5)
print("Hadamard matrix ", 1)
print(H_2_4)
print("Hadamard matrix ", 2)
print(H_2_5)
rho = case.dot(np.transpose(case))
partial_rho = np.array([[rho[0][0]+rho[1][1], rho[0][2]+rho[1][3]],
                        [rho[2][0]+rho[3][1], rho[2][2]+rho[3][3]]])
w, v = LA.eig(partial_rho)
if np.allclose(w, np.array([0.5,0.5])):
    print("Case 2 success")
else:
    print("Case 2 fail")
print(case)
print(w)

# Case 3: 2nd qubit measured 0, 3rd qubit measured 1 and 4th qubit measured 0
p_123 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix-x_matrix)/2), (i_matrix+x_matrix)/2), i_matrix)
h_state3 = p_123.dot(h_state)
h_state3 = h_state3 / LA.norm(h_state3)
case = E_0_8.dot(H_2_3.dot(E_1_16.dot(H_2_4.dot(E_0_32.dot(H_2_5.dot(h_state3))))))

rho = case.dot(np.transpose(case))
partial_rho = np.array([[rho[0][0]+rho[1][1], rho[0][2]+rho[1][3]],
                        [rho[2][0]+rho[3][1], rho[2][2]+rho[3][3]]])
w, v = LA.eig(partial_rho)
if np.allclose(w, np.array([0.5,0.5])):
    print("Case 3 success")
else:
    print("Case 3 fail")
print(case)
print(w)

# Case 4: 2nd qubit measured 0, 3rd qubit measured 1 and 4th qubit measured 1
p_123 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix-x_matrix)/2), (i_matrix-x_matrix)/2), i_matrix)
h_state4 = p_123.dot(h_state)
h_state4 = normalize(h_state4)
case = E_1_8.dot(H_2_3.dot(E_1_16.dot(H_2_4.dot(E_0_32.dot(H_2_5.dot(h_state4))))))

rho = case.dot(np.transpose(case))
partial_rho = np.array([[rho[0][0]+rho[1][1], rho[0][2]+rho[1][3]],
                        [rho[2][0]+rho[3][1], rho[2][2]+rho[3][3]]])
w, v = LA.eig(partial_rho)
if np.allclose(w, np.array([0.5,0.5])):
    print("Case 4 success")
else:
    print("Case 4 fail")
print(w)

# Case 5: 2nd qubit measured 1, 3rd qubit measured 0 and 4th qubit measured 0
p_123 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix-x_matrix)/2), (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), i_matrix)
h_state5 = p_123.dot(h_state)
h_state5 = normalize(h_state5)
case = E_0_8.dot(H_2_3.dot(E_0_16.dot(H_2_4.dot(E_1_32.dot(H_2_5.dot(h_state5))))))

rho = case.dot(np.transpose(case))
partial_rho = np.array([[rho[0][0]+rho[1][1], rho[0][2]+rho[1][3]],
                        [rho[2][0]+rho[3][1], rho[2][2]+rho[3][3]]])
w, v = LA.eig(partial_rho)
if np.allclose(w, np.array([0.5,0.5])):
    print("Case 5 success")
else:
    print("Case 5 fail")
print(case)
print(w)

# Case 6: 2nd qubit measured 1, 3rd qubit measured 0 and 4th qubit measured 1
p_123 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix-x_matrix)/2), (i_matrix+x_matrix)/2), (i_matrix-x_matrix)/2), i_matrix)
h_state6 = p_123.dot(h_state)
h_state6 = normalize(h_state6)
case = E_1_8.dot(H_2_3.dot(E_0_16.dot(H_2_4.dot(E_1_32.dot(H_2_5.dot(h_state6))))))

rho = case.dot(np.transpose(case))
partial_rho = np.array([[rho[0][0]+rho[1][1], rho[0][2]+rho[1][3]],
                        [rho[2][0]+rho[3][1], rho[2][2]+rho[3][3]]])
w, v = LA.eig(partial_rho)
if np.allclose(w, np.array([0.5,0.5])):
    print("Case 6 success")
else:
    print("Case 6 fail")
print(w)

# Case 7: 2nd qubit measured 1, 3rd qubit measured 1 and 4th qubit measured 0
p_123 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix-x_matrix)/2), (i_matrix-x_matrix)/2), (i_matrix-x_matrix)/2), i_matrix)
h_state7 = p_123.dot(h_state)
h_state7 = normalize(h_state7)
case = E_0_8.dot(H_2_3.dot(E_1_16.dot(H_2_4.dot(E_1_32.dot(H_2_5.dot(h_state7))))))

rho = case.dot(np.transpose(case))
partial_rho = np.array([[rho[0][0]+rho[1][1], rho[0][2]+rho[1][3]],
                        [rho[2][0]+rho[3][1], rho[2][2]+rho[3][3]]])
w, v = LA.eig(partial_rho)
if np.allclose(w, np.array([0.5,0.5])):
    print("Case 7 success")
else:
    print("Case 7 fail")
print(w)

# Case 8: 2nd qubit measured 1, 3rd qubit measured 1 and 4th qubit measured 1
p_123 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix-x_matrix)/2), (i_matrix-x_matrix)/2), (i_matrix-x_matrix)/2), i_matrix)
h_state8 = p_123.dot(h_state)
h_state8 = normalize(h_state8)
case = E_1_8.dot(H_2_3.dot(E_1_16.dot(H_2_4.dot(E_1_32.dot(H_2_5.dot(h_state8))))))
rho = case.dot(np.transpose(case))
partial_rho = np.array([[rho[0][0]+rho[1][1], rho[0][2]+rho[1][3]],
                        [rho[2][0]+rho[3][1], rho[2][2]+rho[3][3]]])
w, v = LA.eig(partial_rho)
if np.allclose(w, np.array([0.5,0.5])):
    print("Case 8 success")
else:
    print("Case 8 fail")
print(case)
print(w)




