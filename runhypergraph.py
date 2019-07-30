from hypergraph import *
import math
import itertools

# Find hypergraph states on a set Ω = I + M + O of qubits where there exists Bell type entanglement between
# regions I and O after the qubits in M have been measured in the local X-basis

num_v = 7
possible_faces = []
m = (1, 2, 3,4,5)


def check_each_v_in_a_face(face_list):
    v_set = {0:0, }
    count = 0
    for i in range(1, num_v):
        v_set[i] = 0
    for f in face_list:
        for v in f:
            if v_set[v] == 0:
                count = count + 1
                v_set[v] = 1
    if count == num_v:
        return True
    return False


# create all possible faces given num_v
for face in itertools.combinations(range(0, num_v), 3):
    possible_faces.append(face)


h = Hypergraph(num_v)

# create all possible hypergraphs taking each combination of faces
for i in range(len(possible_faces)):
    i=4
    for faces in itertools.combinations(possible_faces, i+1):
        each_v_in_a_face = True
        each_v_in_a_face = check_each_v_in_a_face(faces)
        if each_v_in_a_face:
            case_list = []
            prob_list = []
            h.construct_state(faces)
            # 1 represents measurement result 1
            # 2 represents measurement result 0
            for case in list(itertools.product([1, 2], repeat=len(m))):
                save_h_state = h.h_state
                m_took = h.perform_measurement(case)
                # probability of measurement case occuring
                prob = np.transpose(save_h_state).dot(m_took.dot(save_h_state))
                h.extract_bits(case)
                rho = h.h_state.dot(np.transpose(h.h_state))
                partial_rho = np.trace(rho.reshape([2, 2, 2, 2]), axis1=1, axis2=3)
                w, v = LA.eig(partial_rho)
                if np.allclose(w, np.array([0.5, 0.5])):
                    case_list.append(case)
                    prob_list.append(prob)
                h.h_state = save_h_state
            if np.isclose(1, sum(prob_list)):
                print("Faces: ", faces, " with cases", case_list, "with total prob", sum(prob_list))
    break











