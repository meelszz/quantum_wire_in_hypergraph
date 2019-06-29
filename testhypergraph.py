from hypergraph import *

# A collection of tests to test functions in the hypergraph class

#utility

# A hypergraph state vector
h_state = 0

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

# Create 5 qubit lantern hypergraph state
h = Hypergraph(((0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 4), (2, 3, 4), (1, 3, 4)), 5)


# Perform on the 5 qubit lantern state:
def run_on_lantern_5():
    h = Hypergraph(((0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 4), (2, 3, 4), (1, 3, 4)), 5)
    h.perform_measurement((1, 2, 3))
    h.h_state = h.h_state / LA.norm(h.h_state)
    h.extract_bits((1, 2, 3))
    h.normalize()
    rho = h.h_state[0].dot(np.transpose(h.h_state[0]))
    partial_rho = np.trace(rho.reshape([2, 2, 2, 2]), axis1=0, axis2=2)
    w, v = LA.eig(partial_rho)
    print(w)


# test creating a hypergraph object
def test_hypergraph_creation():
    h_state_t = np.transpose(h.h_state)

    # Check that the state's inner product is 1
    if 1 != np.all(h_state_t.dot(h.h_state)):
        print("Failed hypergraph creation test 1")

    # Check to see that the state has the proper stabilizers
    cz_23 = i_full - 2 * ((i_full - z_2) / 2).dot((i_full - z_3) / 2)
    cz_34 = i_full - 2 * ((i_full - z_3) / 2).dot((i_full - z_4) / 2)
    cz_24 = i_full - 2 * ((i_full - z_2) / 2).dot((i_full - z_4) / 2)
    cz_13 = i_full - 2 * ((i_full - z_1) / 2).dot((i_full - z_3) / 2)
    cz_14 = i_full - 2 * ((i_full - z_1) / 2).dot((i_full - z_4) / 2)
    cz_35 = i_full - 2 * ((i_full - z_3) / 2).dot((i_full - z_5) / 2)
    cz_45 = i_full - 2 * ((i_full - z_4) / 2).dot((i_full - z_5) / 2)
    s_1 = x_1.dot(cz_23.dot(cz_34.dot(cz_24)))
    s_2 = x_2.dot(cz_13.dot(cz_14.dot(cz_35.dot(cz_45))))

    test2 = h_state_t.dot(s_1.dot(h.h_state))
    test3 = h_state_t.dot(s_2.dot(h.h_state))

    if not test2:
        print("Failed hypergraph creation test 2")

    if not test3:
        print("Failed hypergraph creation test 3")

    # Compare with correctly constructed 5 qubit graph state
    cz_123 = i_full - 2 * ((i_full - z_1) / 2).dot((i_full - z_2) / 2).dot(((i_full - z_3) / 2))
    cz_134 = i_full - 2 * ((i_full - z_1) / 2).dot((i_full - z_3) / 2).dot((i_full - z_4) / 2)
    cz_124 = i_full - 2 * ((i_full - z_1) / 2).dot((i_full - z_2) / 2).dot((i_full - z_4) / 2)
    cz_235 = i_full - 2 * ((i_full - z_2) / 2).dot((i_full - z_3) / 2).dot((i_full - z_5) / 2)
    cz_345 = i_full - 2 * ((i_full - z_3) / 2).dot((i_full - z_4) / 2).dot((i_full - z_5) / 2)
    cz_245 = i_full - 2 * ((i_full - z_2) / 2).dot((i_full - z_4) / 2).dot((i_full - z_5) / 2)

    h_built = cz_123.dot(cz_134.dot(cz_124.dot(cz_235.dot(cz_345.dot(cz_245.dot(plus))))))

    test4 = np.array_equal(h_built, h.h_state)

    if not test4:
        print("Failed hypergraph creation test 1")


# test perform measurement
def test_perform_measurement():

    p_123 = np.kron(np.kron(np.kron(np.kron(i_matrix, (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), (i_matrix+x_matrix)/2), i_matrix)
    after_m = p_123.dot(h.h_state)
    after_m = after_m / LA.norm(after_m)

    h.perform_measurement((1, 2, 3))

    if not np.array_equal(after_m, h.h_state):
        print("Failed perform measurement")
    else:
        print("we gucc")

    # Check normalization
    if 1 != np.transpose(h.h_state).dot(h.h_state):
        print("Normalization after measurement failed")


# Do an explicit extraction of qubits 1 and 5
def explicit_extraction():
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
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                        0]])

    # hadamard on qubit 1 acting on 5
    hadamard_1_5 = np.kron(np.kron(np.kron(np.kron(i_matrix, h_matrix), i_matrix), i_matrix), i_matrix)
    h_state = elim_2.dot(hadamard_1_5.dot(h.h_state))

    # print("First:",h_state)

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
    # print(w)


# Run tests on hypergraph functions
test_hypergraph_creation()
test_perform_measurement()

# Some tests with partial density matrices
a = np.array([[1], [0]])
b = np.array([[0], [1]])
state = np.kron(a,b)
rho = state.dot(np.transpose(state))
partial_rho = np.trace(rho.reshape([2, 2, 2, 2]), axis1=1, axis2=3)
#print(partial_rho)
#print(a.dot(np.transpose(a)))
#print(b.dot(np.transpose(b)))

