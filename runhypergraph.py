from hypergraph import *
import itertools
import time

# Find hypergraph states on a set Ω = I + M + O of qubits where there exists Bell type entanglement between
# regions I and O after the qubits in M have been measured in the local X-basis

start_time = time.time()

# num_v is number of qubits (vertices in hypergrah)
num_v = 7
possible_faces = []
# m is the measurement set
m = (1, 2, 3, 4, 5)

# create all possible faces given num_v
for face in itertools.combinations(range(0, num_v), 3):
    possible_faces.append(face)

# create utility for hypergraph with num_v number of qubits
# create all Z, I, Projection(for measurement), Extraction and Hadamard matrices

h = Hypergraph(num_v)

k = 0

time_1 = time.perf_counter()
# create all possible hypergraphs taking each combination of faces
for i in range(len(possible_faces)):
    i = 18
    for faces in itertools.combinations(possible_faces, i+1):
        k += 1
        case_list = []
        prob_list = []
        time_3 = time.perf_counter()
        h.construct_state(faces)
        time_4 = time.perf_counter()
        # case[k] == 0 represents measurement result of m[k] == 0
        # case[k] == 1 represents measurement result of m[k] == 1
        for case in list(itertools.product([0, 1], repeat=len(m))):
            save_h_state = h.h_state
            # m_took is the measurement operator used on the state (saved to find prob)
            m_took = h.perform_measurement(case)
            # probability of measurement case occurring
            prob = np.transpose(save_h_state).dot(m_took.dot(save_h_state))
            # extract the bits that were measured
            h.extract_bits(case)

            # check if resulting qubits are maximally entangled using density matrix
            rho = h.h_state.dot(np.transpose(h.h_state))
            partial_rho = np.trace(rho.reshape([2, 2, 2, 2]), axis1=1, axis2=3)
            w, v = LA.eig(partial_rho)
            if np.allclose(w, np.array([0.5, 0.5])):
                case_list.append(case)
                prob_list.append(prob)
            # revert h_state back to its pre-measurement state to try the next case
            h.h_state = save_h_state
        if np.isclose(1, sum(prob_list)):
            print("Faces: ", faces, " with cases", case_list, "with total prob", sum(prob_list))
        if k == 1:
            break
    if k == 1:
        break


time_2 = time.perf_counter()
end_time = time.time()

#print("Total time:", end_time - start_time)
print("Time 1:", time_2 - time_1)
print("Time 2:", time_4 - time_3)

print(len(list(itertools.product([0, 1], repeat=4))))






