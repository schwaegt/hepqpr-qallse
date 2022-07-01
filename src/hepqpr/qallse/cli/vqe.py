import numpy as np
from numpy.random import random_sample
import matplotlib.pyplot as plt
import pylab
from qiskit import Aer
from qiskit import IBMQ
from qiskit import QuantumCircuit
from qiskit.opflow import X, Z, I, CircuitSampler, ExpectationFactory, PauliExpectation, CircuitStateFn, StateFn
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SPSA, QNSPSA, NFT
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import EfficientSU2
from qiskit.providers.aer.noise import NoiseModel, ReadoutError
import os
from os import listdir
from os.path import isfile, join
import ast
import pickle
import argparse
from math import sqrt
import threading
import concurrent.futures
from copy import copy
import random


def linear_qubo(Q):

    '''Extract linear terms of QUBO. Including terms with weight 0 explicitly'''

    Q_linear = {}
    triplets = []
    for key, item in Q.items():
        if key[0] == key[1]:
            Q_linear[key] = item
            triplets.append(key[0])

    for key in Q:
        if key[0] not in triplets:
            Q_linear[(key[0], key[0])] = 0.
            triplets.append(key[0])
        if key[1] not in triplets:
            Q_linear[(key[1], key[1])] = 0.
            triplets.append(key[0])

    return Q_linear


def max_rz_angle(qubo_entry, xplets):

    rz_t1_d1 = xplets[xplets[qubo_entry[0][0]]['d1']]['rz_angle']
    rz_t1_d2 = xplets[xplets[qubo_entry[0][0]]['d2']]['rz_angle']
    rz_t2_d1 = xplets[xplets[qubo_entry[0][1]]['d1']]['rz_angle']
    rz_t2_d2 = xplets[xplets[qubo_entry[0][1]]['d2']]['rz_angle']

    return max(rz_t1_d1, rz_t1_d2, rz_t2_d1 ,rz_t2_d2)


def energy_bit_flip(state, triplet_id, relations, H):

    en = H.eval(state).eval(state)
    state_list = list(state)
    index = relations[triplet_id]
    if state_list[index] == '0':
        state_list[index] = '1'
    elif state_list[index] == '1':
        state_list[index] = '0'
    state_flipped = ''.join(state_list)
    en_flipped = H.eval(state_flipped).eval(state_flipped)
    en_diff = abs(en_flipped - en)
    print('flip')
    return en_diff


def slice_qubo(Q, xplets, size):

    '''Split QUBO into sub-QUBOs. Implementation not efficient !'''

    Q_linear = linear_qubo(Q)
    Q_linear_list = sorted(Q_linear.items(), key = lambda qubo_entry: max_rz_angle(qubo_entry, xplets))
    Q_linear_slices = [dict(Q_linear_list[i*size:(i+1)*size]) for i in range(len(Q_linear_list)//size)]
    if len(Q_linear_list) % size != 0:
        Q_linear_slices.append(dict(Q_linear_list[-(len(Q_linear_list) % size):]))

    Q_slices = []
    for count, Q_linear_slice in enumerate(Q_linear_slices):
        triplets = [x[0] for x in Q_linear_slice.keys()]
        Q_slice = {}
        for key, item in Q.items():
            if key[0] in triplets and key[1] in triplets:
                Q_slice[(key[0], key[1])] = item
        Q_slice_tupel = (count, Q_slice)
        Q_slices.append(Q_slice_tupel)

    return Q_slices


def create_sub_qubos(Q, H, relations, xplets, size, state):

    '''Split QUBO into sub-QUBOs acoording to impact of bit flip on energy'''

    Q_linear = linear_qubo(Q)
    Q_linear_list = sorted(Q_linear.items(), key = lambda qubo_entry: - energy_bit_flip(state, qubo_entry[0][0], relations, H))
    Q_linear_slices = [dict(Q_linear_list[i*size:(i+1)*size]) for i in range(len(Q_linear_list)//size)]
    if len(Q_linear_list) % size != 0:
        Q_linear_slices.append(dict(Q_linear_list[-(len(Q_linear_list) % size):]))

    Q_slices = []
    for count, Q_linear_slice in enumerate(Q_linear_slices):
        triplets = [x[0] for x in Q_linear_slice.keys()]
        Q_slice = {}
        for key, item in Q.items():
            if key[0] in triplets and key[1] in triplets:
                Q_slice[(key[0], key[1])] = item
        Q_slice_tupel = (count, Q_slice)
        Q_slices.append(Q_slice_tupel)

    return Q_slices


def prepare_data_dicts(data):

    '''Translate input data into iterable dictionaries'''

    b_ij = {}
    a_i = {}
    relations = {}
    k = 0

    for key in data:
        if key[1] in relations:
            j = relations[key[1]]

        else:
            j = k
            relations[key[1]] = j
            k += 1

        if key[0] in relations:
            i = relations[key[0]]

        else:
            i = k
            relations[key[0]] = i
            k += 1

        if i > j:
            b_ij.update({(i, j) : data[(key[0], key[1])]})

        elif i < j:
            b_ij.update({(j, i) : data[(key[0], key[1])]})

        elif i == j:
            a_i.update({i : data[(key[0], key[1])]})

    nqubits = len(relations)

    for i in range(nqubits):
        for j in range(i):
            if (i, j) not in b_ij:
                b_ij[(i, j)] = 0

    for i in range(nqubits):
        if i not in a_i:
            a_i[i] = 0

    return b_ij, a_i, relations


def Tracking_Hamiltonian(b_ij, a_i):

    '''Given coupling strenghts b_ij and bias weights a_i return the tracking Hamiltonian as a Qiskit PauliOp object'''

    nqubits = len(a_i)

    H = I - I
    H = H^nqubits

    #prepare quadratic term
    for i in range(nqubits):
        for j in range(i):

            n_left = nqubits - i - 1
            n_middle = i - j - 1
            n_right = j

            temp = Z

            if n_left > 0:

                id_left = I^n_left
                temp = id_left^temp

            if n_middle > 0:

                id_middle = I^n_middle
                temp = temp^id_middle

            temp = temp^Z

            if n_right > 0:

                id_right = I^n_right
                temp = temp^id_right

            H += b_ij[(i, j)] * temp

    #prepare linear term
    for i in range(nqubits):

        bias = 0
        for j in range(nqubits):
            if j<i:
                bias += b_ij[(i,j)]
            if i<j:
                bias += b_ij[(j,i)]

        bias += 2*a_i[i]

        n_left = nqubits - i - 1
        n_right = i

        temp = Z

        if n_left > 0:

            id_left = I^n_left
            temp = id_left^temp

        if n_right > 0:

            id_right = I^n_right
            temp = temp^id_right

        H += -1.0 * bias * temp

    return H


def construct_rotation_layer(n_qubits, gate, params):

    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        if gate == 'rx':
            qc.rx(params[i], i)

        elif gate == 'ry':
            qc.ry(params[i], i)

        elif gate == 'rz':
            qc.rz(params[i], i)

    return qc


def return_optimizer(optimizer_name, maxiter):

    if optimizer_name=='SPSA':
        optimizer=SPSA(maxiter=maxiter)
    elif optimizer_name=='COBYLA':
        optimizer=COBYLA(maxiter=maxiter)
    elif optimizer_name=='NFT':
        optimizer=NFT(maxiter=maxiter)

    return optimizer


def construct_entanglement_layer(n_qubits, entanglement, inverse=False):

    qc = QuantumCircuit(n_qubits)

    if inverse==False:
        if entanglement == 'full':
            for i in range(n_qubits):
                for j in range(i):
                    qc.cnot(j, i)

        elif entanglement == 'linear':
            for i in range(n_qubits-1):
                qc.cnot(i,i+1)

        elif entanglement == None:
            pass

    elif inverse==True:
        if entanglement == 'full':
            for i in reversed(range(n_qubits)):
                for j in range(i):
                    qc.cnot(j, i)

        elif entanglement == 'linear':
            for i in reversed(range(n_qubits-1)):
                qc.cnot(i,i+1)

        elif entanglement == None:
            pass

    return qc


def translate_vqe_result(result, relations):

    result_translated = {}
    relations_inv = {v: k for k, v in relations.items()}
    counts = result['eigenstate']
    key_max = max(counts, key = lambda x: counts[x])
    #reverse string because of qiskit convention for counting qubits
    key_max_reverse = key_max[::-1]
    for count, value in enumerate(key_max_reverse):
        result_translated[relations_inv[count]]=int(value)

    return result_translated


def solve_vqe_slices(Q_slices, solver_config):

    n_slices = len(Q_slices)

    result_full = []

    global lock
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_slices+1) as executor_0:
         for result_ten_runs_tupel in executor_0.map(lambda Q: solve_vqe_ten(Q, solver_config = solver_config), Q_slices):
            result_full.append(result_ten_runs_tupel)

    return result_full


def solve_vqe_sub_qubos(Q, xplets, **kwargs):

    b_ij, a_i, relations = prepare_data_dicts(Q)
    n_triplets = len(relations)
    H = Tracking_Hamiltonian(b_ij, a_i)
    state = ''.join(random.choices(['0', '1'], k = n_triplets))
    size = kwargs['sub_qubo_size']

    for i in range(kwargs['sub_qubo_iterations']):
        result_full = []
        Q_slices = create_sub_qubos(Q, H, relations, xplets, size, state)
        n_slices = len(Q_slices)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_slices+1) as executor:
             for result_ten_runs_tupel in executor.map(lambda Q: solve_vqe_ten(Q, solver_config = solver_config), Q_slices):
                result_full.append(result_ten_runs_tupel)

        state_list = list(state)

        for result_ten_runs_tupel in result_full:
            result_ten_runs = result_ten_runs_tupel[1]
            result_lowest_energy = min(result_ten_runs, key=lambda result_one_run: result_one_run['optimal_value'])

            for triplet_id, value in result_lowest_energy['eigenstate_translated'].items():
                index = relations[triplet_id]
                state_list[index] = str(value)

        state = ''.join(state_list)

    return result_full


def solve_eigensolver_sub_qubos(Q, xplets, **kwargs):

    b_ij, a_i, relations = prepare_data_dicts(Q)
    n_triplets = len(relations)
    H = Tracking_Hamiltonian(b_ij, a_i)
    state = ''.join(random.choices(['0', '1'], k = n_triplets))
    size = kwargs['sub_qubo_size']

    for i in range(kwargs['sub_qubo_iterations']):
        Q_slices = create_sub_qubos(Q, H, relations, xplets, size, state)
        n_slices = len(Q_slices)
        results = solve_eigensolver_slices(Q_slices)

        state_list = list(state)

        for result_tupel in results:
                result = result_tupel[1]

                for triplet_id, value in result['eigenstate_translated'].items():
                    index = relations[triplet_id]
                    state_list[index] = str(value)

        state = ''.join(state_list)

    return results


def solve_vqe_ten(Q_slice, **kwargs):

    try:

        slice = Q_slice[0]
        Q = Q_slice[1]
        b_ij, a_i, relations = prepare_data_dicts(Q)
        op = Tracking_Hamiltonian(b_ij, a_i)
        n_qubits = len(relations)
        params = ParameterVector('params', n_qubits)
        ansatz = construct_rotation_layer(n_qubits, 'ry', params[0:n_qubits])
        optimizer = return_optimizer(kwargs['optimizer_name'], kwargs['maxiter'])
        options = {'backend_name': kwargs['backend_name']}
        runtime_inputs = {
        'ansatz': ansatz,
        'aux_operators': None,
        'initial_layout': None,
        'initial_parameters': None,
        'measurement_error_mitigation': None,
        'operator': op,
        'optimizer': optimizer,
        'shots': kwargs['shots']
        }
        result_ten_runs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=11) as executor_1:
            for result_one_run in executor_1.map(lambda run: solve_vqe_one(run, options, runtime_inputs, n_qubits, relations), [i for i in range(10)]):
                result_translated = translate_vqe_result(result_one_run, relations)
                result_one_run.update({'eigenstate_translated': result_translated})
                with lock:
                    result_ten_runs.append(result_one_run)

        result_ten_runs_tupel = (slice, result_ten_runs)

        return result_ten_runs_tupel

    except:

        print('A full sclice failed')



def solve_vqe_one(run, options, runtime_inputs, n_qubits, relations):

    try:

        with lock:
            runtime_inputs_local = copy(runtime_inputs)

        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q-desy', group='internal', project='tracking')

        initial_point = [2 * np.pi * x for x in random_sample(n_qubits)]
        runtime_inputs_local.update({'initial_parameters': initial_point})

        job = provider.runtime.run(program_id='vqe',options=options,inputs=runtime_inputs_local)
        result = job.result()

        energy = result['optimal_value']
        result_translated = translate_vqe_result(result, relations)

        return result

    except:

        print('A single run failed')


def solve_eigensolver_slices(Q_slices):

    n_slices = len(Q_slices)
    results = []

    for Q_slice in Q_slices:
        slice = Q_slice[0]
        Q = Q_slice[1]
        result_dict = {}
        b_ij, a_i, relations = prepare_data_dicts(Q)
        op = Tracking_Hamiltonian(b_ij, a_i)
        npme = NumPyMinimumEigensolver()
        result_eigensolver = npme.compute_minimum_eigenvalue(operator=op)
        counts_eigensolver = result_eigensolver.eigenstate.to_dict_fn().sample()
        result_dict.update({'eigenstate': counts_eigensolver})
        result_dict.update({'optimal_value': result_eigensolver.eigenvalue})
        result_translated = translate_vqe_result(result_dict, relations)
        result_dict.update({'eigenstate_translated': result_translated})
        result_tupel = (slice, result_dict)

        results.append(result_tupel)
        print('solved slice '+ str(slice+1) + ' of ' + str(n_slices))

    return results
