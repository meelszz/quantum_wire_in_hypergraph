from hypergraph import *
import numpy as np

h = Hypergraph(((0,1,2),(0,2,3),(0,3,4),(0,4,5),(0,5,1),(6,1,2),(6,2,3),(6,3,4),(6,4,5),(6,5,1)), 7)

# Test probability of outcome (+,-,-,+,+), (-,+,-,+,+) and  (-,-,+,+,+) as measurement of qubits 2,3,4,5,6 in 7 qubit lantern state
# (+,-,-,+,+)
x_23456 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix-x_matrix)/2),(i_matrix-x_matrix)/2),(i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), i_matrix)
prob = np.transpose(h.h_state).dot(x_23456.dot(h.h_state))
#print("Prob of measurement of qubits 2,3,4,5,6 as (-,+,-,+,+):", prob)

# (-,+,-,+,+)
x_23456 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix-x_matrix)/2), (i_matrix+x_matrix)/2),(i_matrix-x_matrix)/2),(i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), i_matrix)
prob = np.transpose(h.h_state).dot(x_23456.dot(h.h_state))
#print("Prob of measurement of qubits 2,3,4,5,6 as (-,+,-,+,+):", prob)

# (-,-,+,+,+)
x_23456 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix-x_matrix)/2), (i_matrix-x_matrix)/2),(i_matrix+x_matrix)/2),(i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), i_matrix)
prob = np.transpose(h.h_state).dot(x_23456.dot(h.h_state))
#print("Prob of measurement of qubits 2,3,4,5,6 as (-,-,+,+,+):", prob)


