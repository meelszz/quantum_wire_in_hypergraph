from hypergraph import *
import math
import itertools

# Find hypergraph states on a set Ω = I + M + O of qubits where there exists Bell type entanglement between
# regions I and O after the qubits in M have been measured in the local X-basis

num_v = 5
possible_faces = []
possible_measurements = []

# create all possible faces given num_v
for face in itertools.combinations(range(0, num_v), 3):
    possible_faces.append(face)

for m in itertools.combinations(range(0, num_v), num_v-2):
    possible_measurements.append(m)

# create all possible hypergraphs taking each combination of faces
for i in range(0, len(possible_faces)):
    # print(i)
    for faces in itertools.combinations(possible_faces, i+1):
        each_v_in_a_face = True
        for v in range(num_v):
            v_in_face = False
            for f in faces:
                if v in f:
                    v_in_face = True
            if not v_in_face:
                each_v_in_a_face = False

        if each_v_in_a_face:
            for m in possible_measurements:
                h = Hypergraph(faces, num_v)
                h.perform_measurement(m)
                h.extract_bits(m)
                h.normalize()
                rhos = []
                for state in h.h_state:
                    # find density matrix
                    rhos.append(state.dot(np.transpose(state)))
                successes = 0
                for rho in rhos:
                    partial_rho = np.trace(rho.reshape([2, 2, 2, 2]), axis1=1, axis2=3)
                    w, v = LA.eig(partial_rho)
                    if np.allclose(w, np.array([0.5,0.5])):
                        successes = successes + 1
                if successes > 0:
                    print("Faces: ", faces, "with measurement", m, "and works with", successes, "m results")


# Perform explicitly on 5 qubit lantern state:

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

elim_2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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

print(h_state)

#rho = h_state.dot(np.transpose(h_state))
#partial_rho = np.trace(rho.reshape([2, 2, 2, 2]), axis1=0, axis2=2)
#w, v = LA.eig(partial_rho)
#print(w)

