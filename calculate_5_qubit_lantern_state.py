import numpy as np
from numpy import linalg as LA

plus_state = np.array([[np.sqrt(.5)],
                       [np.sqrt(.5)]])
i_matrix = np.array([[1, 0],
                     [0, 1]])
z_matrix = np.array([[1, 0],
                     [0, -1]])
x_matrix = np.array([[0, 1],
                     [1, 0]])

h_matrix = (np.sqrt(.5))*np.array([[1,1],
                                   [1,-1]])

# 5 qubit state in the Pauli-X basis
plus = np.kron(np.kron(np.kron(np.kron(plus_state, plus_state), plus_state), plus_state), plus_state)

# 5 qubit identity matrix
i_full = np.kron(np.kron(np.kron(np.kron(i_matrix, i_matrix), i_matrix), i_matrix), i_matrix)

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

# Save pre-measurement hypergraph state
save_h_state = h_state

# Perform measurement and normalize
p_123 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), i_matrix)
h_state = p_123.dot(h_state)
h_state = h_state / LA.norm(h_state)

# Extract qubits 1 and 5 under diff measurment outcomes

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

# Case 1: 2nd qubit measured 0, 3rd qubit measured 0 and 4th qubit measured 0
case = E_0_8.dot(H_2_3.dot(E_0_16.dot(H_2_4.dot(E_0_32.dot(H_2_5.dot(h_state))))))

rho = case.dot(np.transpose(case))
partial_rho = np.array([[rho[0][0]+rho[1][1], rho[0][2]+rho[1][3]],
                        [rho[2][0]+rho[3][1], rho[2][2]+rho[3][3]]])
w, v = LA.eig(partial_rho)
if np.allclose(w, np.array([0.5,0.5])):
    print("Case 1 success")
else:
    print("Case 1 fail")
print(w)

# Case 2: 2nd qubit measured 0, 3rd qubit measured 0 and 4th qubit measured 1
case = E_1_8.dot(H_2_3.dot(E_0_16.dot(H_2_4.dot(E_0_32.dot(H_2_5.dot(h_state))))))

rho = case.dot(np.transpose(case))
partial_rho = np.array([[rho[0][0]+rho[1][1], rho[0][2]+rho[1][3]],
                        [rho[2][0]+rho[3][1], rho[2][2]+rho[3][3]]])
w, v = LA.eig(partial_rho)
if np.allclose(w, np.array([0.5,0.5])):
    print("Case 2 success")
else:
    print("Case 2 fail")
print(w)

# Case 3: 2nd qubit measured 0, 3rd qubit measured 1 and 4th qubit measured 0
case = E_0_8.dot(H_2_3.dot(E_1_16.dot(H_2_4.dot(E_0_32.dot(H_2_5.dot(h_state))))))

rho = case.dot(np.transpose(case))
partial_rho = np.array([[rho[0][0]+rho[1][1], rho[0][2]+rho[1][3]],
                        [rho[2][0]+rho[3][1], rho[2][2]+rho[3][3]]])
w, v = LA.eig(partial_rho)
if np.allclose(w, np.array([0.5,0.5])):
    print("Case 3 success")
else:
    print("Case 3 fail")
print(w)

# Case 4: 2nd qubit measured 0, 3rd qubit measured 1 and 4th qubit measured 1
case = E_1_8.dot(H_2_3.dot(E_1_16.dot(H_2_4.dot(E_0_32.dot(H_2_5.dot(h_state))))))

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
case = E_0_8.dot(H_2_3.dot(E_0_16.dot(H_2_4.dot(E_1_32.dot(H_2_5.dot(h_state))))))

rho = case.dot(np.transpose(case))
partial_rho = np.array([[rho[0][0]+rho[1][1], rho[0][2]+rho[1][3]],
                        [rho[2][0]+rho[3][1], rho[2][2]+rho[3][3]]])
w, v = LA.eig(partial_rho)
if np.allclose(w, np.array([0.5,0.5])):
    print("Case 5 success")
else:
    print("Case 5 fail")
print(w)

# Case 6: 2nd qubit measured 1, 3rd qubit measured 0 and 4th qubit measured 1
case = E_1_8.dot(H_2_3.dot(E_0_16.dot(H_2_4.dot(E_1_32.dot(H_2_5.dot(h_state))))))

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
case = E_0_8.dot(H_2_3.dot(E_1_16.dot(H_2_4.dot(E_1_32.dot(H_2_5.dot(h_state))))))

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
case = E_1_8.dot(H_2_3.dot(E_1_16.dot(H_2_4.dot(E_1_32.dot(H_2_5.dot(h_state))))))
rho = case.dot(np.transpose(case))
partial_rho = np.array([[rho[0][0]+rho[1][1], rho[0][2]+rho[1][3]],
                        [rho[2][0]+rho[3][1], rho[2][2]+rho[3][3]]])
w, v = LA.eig(partial_rho)
if np.allclose(w, np.array([0.5,0.5])):
    print("Case 8 success")
else:
    print("Case 8 fail")
print(w)


# Test that any extraction matrix with assumed result of |1> takes state to all 0:
if 0 == np.count_nonzero(E_1_32.dot(H_2_5.dot(h_state))):
    print("Test 1 pass")
else:
    print("Test 1 fail")

if 0 != np.count_nonzero((E_0_32.dot(H_2_5.dot(h_state)))):
    print("Test 2 pass")
else:
    print("Test 2 fail")

if 0 == np.count_nonzero(E_1_16.dot(H_2_4.dot(E_0_32.dot(H_2_5.dot(h_state))))):
    print("Test 3 pass")
else:
    print("Test 3 fail")

if 0 != np.count_nonzero((E_0_16.dot(H_2_4.dot(E_0_32.dot(H_2_5.dot(h_state)))))):
    print("Test 4 pass")
else:
    print("Test 4 fail")

if 0 == np.count_nonzero(E_1_8.dot(H_2_3.dot(E_0_16.dot(H_2_4.dot(E_0_32.dot(H_2_5.dot(h_state))))))):
    print("Test 5 pass")
else:
    print("Test 5 fail")



