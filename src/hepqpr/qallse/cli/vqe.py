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
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_optimization import QuadraticProgram
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


def qubo_from_linear(Q_full, Q_linear):

    '''complete linear qubo with quadratic terms'''

    Q = {}
    triplets = [x[0] for x in Q_linear.keys()]

    for key, value in Q_full.items():
        if key[0] in triplets and key[1] in triplets:
            Q[(key[0], key[1])] = value

    return Q


def reduce_qubo(qubo):

    '''Implements QUBO preprocessing of arxiv 1705.09844.
        Neglect constant shift of objective function'''

    vars_all = {}
    vars_determined = {}
    sums_positive_offdiag = {}
    sums_negative_offdiag = {}
    #Paper defines preprocessing for maximizing
    qubo = {key: -value for key, value in qubo.items()}
    #create dict for triplets and compute off-diagonal sums
    for key, value in qubo.items():
        if key[0] not in vars_all:
            vars_all.update({key[0]: -1})

        if key[1] not in vars_all:
            vars_all.update({key[1]: -1})

        if key[0] != key[1]:
            if value > 0:
                if key[0] not in sums_positive_offdiag:
                    sums_positive_offdiag.update({key[0]: 0.})

                if key[1] not in sums_positive_offdiag:
                    sums_positive_offdiag.update({key[1]: 0.})

                sums_positive_offdiag[key[0]] += value
                sums_positive_offdiag[key[1]] += value

            if value < 0:
                if key[0] not in sums_negative_offdiag:
                    sums_negative_offdiag.update({key[0]: 0.})

                if key[1] not in sums_negative_offdiag:
                    sums_negative_offdiag.update({key[1]: 0.})

                sums_negative_offdiag[key[0]] += value
                sums_negative_offdiag[key[1]] += value

    var_determined = True

    while var_determined:
        var_determined = False
        vars_all_copy = copy(vars_all)

        for var_i in vars_all_copy:
            qubo.get((var_i, var_i), 0.)
            #rule_1
            if qubo.get((var_i, var_i), 0.) + sums_negative_offdiag.get(var_i, 0.) >= 0:
                var_determined = True
                vars_determined.update({var_i: 1})
                vars_all.pop(var_i)
                #update diagonal coefficients
                temp = {key: value for key, value in qubo.items() if (key[0] == var_i) ^ (key[1] == var_i)}
                temp_sym = {(key[1], key[0]): value for key, value in temp.items()}
                temp.update(temp_sym)
                qubo = {key: qubo.get(key, 0.) + temp.get((var_i, key[0]), 0.) for key in qubo if key[0] == key[1]}
                #update off diagonal sums
                temp_negative = {key[0]: sum(v for k, v in temp.items() if k[0] == key[0] and v < 0) for key in temp}
                temp_positive = {key[0]: sum(v for k, v in temp.items() if k[0] == key[0] and v > 0) for key in temp}
                sums_negative_offdiag = {key: sums_negative_offdiag[key] - temp_negative[key] for key in temp_negative}
                sums_positive_offdiag = {key: sums_positive_offdiag[key] - temp_positive[key] for key in temp_positive}
                #remove row_var and column_var from qubo
                temp.update({(var_i, var_i): qubo.get((var_i, var_i), 0.)})
                qubo = {key: value for key, value in qubo.items() if key not in temp}
                #neglect constant shift of objective function
                break
            #rule_2
            elif qubo.get((var_i, var_i), 0.) + sums_positive_offdiag.get(var_i, 0.) <= 0:
                var_determined = True
                vars_determined.update({var_i: 0})
                vars_all.pop(var_i)
                #no update of diagonal coefficients
                #remove row_var and column_var from qubo
                temp = {key: value for key, value in qubo.items() if (key[0] == var_i) ^ (key[1] == var_i)}
                temp_sym = {(key[1], key[0]): value for key, value in temp.items()}
                temp.update(temp_sym)
                temp.update({(var_i, var_i): qubo.get((var_i, var_i), 0.)})
                qubo = {key: value for key, value in qubo.items() if key not in temp}
                #no adjustment of objective function
                break

        vars_all_copy = copy(vars_all)
        #rule_3
        for var_i in vars_all_copy:
            for var_h in vars_all_copy:
                if (var_i != var_h) and (qubo.get((var_i, var_h), 0.) > 0) and (qubo.get((var_i, var_i), 0.) + qubo.get((var_h, var_h), 0.) + qubo.get((var_i, var_h), 0.) + sums_negative_offdiag.get(var_i, 0.) + sums_negative_offdiag.get(var_h, 0.)) >= 0:
                    var_determined = True
                    vars_determined.update({var_i: 1, var_h: 1})
                    vars_all.pop(var_i)
                    vars_all.pop(var_h)
                    #not sure if remove from qubo
                    temp = {key: value for key, value in qubo.items() if (key[0] == var_i) ^ (key[1] == var_i)}
                    temp_sym = {(key[1], key[0]): value for key, value in temp.items()}
                    temp.update(temp_sym)
                    temp.update({(var_i, var_i): qubo.get((var_i, var_i), 0.)})
                    qubo = {key: value for key, value in qubo.items() if key not in temp}
                    temp = {key: value for key, value in qubo.items() if (key[0] == var_h) ^ (key[1] == var_h)}
                    temp_sym = {(key[1], key[0]): value for key, value in temp.items()}
                    temp.update(temp_sym)
                    temp.update({(var_h, var_h): qubo.get((var_h, var_h), 0.)})
                    qubo = {key: value for key, value in qubo.items() if key not in temp}
                    break

                if var_determined:
                    break

    qubo = {key: -value for key, value in qubo.items()}

    return qubo, vars_determined


def max_rz_angle(qubo_entry, xplets):

    rz_t1_d1 = xplets[xplets[qubo_entry[0][0]]['d1']]['rz_angle']
    rz_t1_d2 = xplets[xplets[qubo_entry[0][0]]['d2']]['rz_angle']
    rz_t2_d1 = xplets[xplets[qubo_entry[0][1]]['d1']]['rz_angle']
    rz_t2_d2 = xplets[xplets[qubo_entry[0][1]]['d2']]['rz_angle']

    return max(rz_t1_d1, rz_t1_d2, rz_t2_d1, rz_t2_d2)


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
    return en_diff


def slice_qubo(Q, xplets, size, overlap):

    '''Split QUBO into sub-QUBOs. Implementation not efficient !'''

    Q_linear = linear_qubo(Q)
    Q_linear_list = sorted(Q_linear.items(), key = lambda qubo_entry: max_rz_angle(qubo_entry, xplets))
    Q_linear_slices = [dict(Q_linear_list[i:i+size]) for i in range(0, len(Q_linear_list), size-overlap)]
    Q_slices = []

    for Q_linear_slice in Q_linear_slices:
        triplets = [x[0] for x in Q_linear_slice.keys()]
        Q_slice = {}

        for key, item in Q.items():
            if key[0] in triplets and key[1] in triplets:
                Q_slice[(key[0], key[1])] = item

        Q_slices.append(Q_slice)

    return Q_slices


def create_sub_qubos(Q, H, relations, xplets, size, state):

    '''Split QUBO into sub-QUBOs acoording to impact of bit flip on energy'''

    Q_linear = linear_qubo(Q)
    Q_linear_list = sorted(Q_linear.items(), key = lambda qubo_entry: energy_bit_flip(state, qubo_entry[0][0], relations, H))
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


# def prepare_data_dicts(data):
#
#     '''Translate input data into iterable dictionaries'''
#
#     b_ij = {}
#     a_i = {}
#     relations = {}
#     k = 0
#
#     for key in data:
#         if key[1] in relations:
#             j = relations[key[1]]
#
#         else:
#             j = k
#             relations[key[1]] = j
#             k += 1
#
#         if key[0] in relations:
#             i = relations[key[0]]
#
#         else:
#             i = k
#             relations[key[0]] = i
#             k += 1
#
#         if i > j:
#             b_ij.update({(i, j) : data[(key[0], key[1])]})
#
#         elif i < j:
#             b_ij.update({(j, i) : data[(key[0], key[1])]})
#
#         elif i == j:
#             a_i.update({i : data[(key[0], key[1])]})
#
#     nqubits = len(relations)
#
#     for i in range(nqubits):
#         for j in range(i):
#             if (i, j) not in b_ij:
#                 b_ij[(i, j)] = 0
#
#     for i in range(nqubits):
#         if i not in a_i:
#             a_i[i] = 0
#
#     return b_ij, a_i, relations


def tracking_hamiltonian(Q):

    '''Build Tracking Hamiltonian directly from QUBO'''

    Q_linear = linear_qubo(Q)
    Q_full = qubo_from_linear(Q, Q_linear)
    triplets = [x[0] for x in Q_linear.keys()]
    n_triplets = len(triplets)
    b_sums = {triplet_id: 0. for triplet_id in triplets}
    H = I - I
    H = H^n_triplets
    relations = {}
    #prepare quadratic term first because we need all b_ij for the linear term
    k = 0
    for key, value in Q_full.items():
        if key[0] != key[1]:
            if key[0] not in relations:
                relations[key[0]] = k
                k += 1

            if key[1] not in relations:
                relations[key[1]] = k
                k += 1

            b_sums[key[0]] += value
            b_sums[key[1]] += value

            i = max(relations[key[0]], relations[key[1]])
            j = min(relations[key[0]], relations[key[1]])

            n_left = n_triplets - i - 1
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

            H += value * temp
        #prepare linear term
    for key, value in Q_full.items():
        if key[0] == key[1]:

            if key[0] not in relations:
                relations[key[0]] =  k
                k += 1

            i = relations[key[0]]

            n_left = n_triplets - i - 1
            n_right = i

            temp = Z

            if n_left > 0:

                id_left = I^n_left
                temp = id_left^temp

            if n_right > 0:

                id_right = I^n_right
                temp = temp^id_right

            H += -1. * (b_sums[key[0]] + 2. * value) * temp

    return H, relations


# def tracking_hamiltonian(Q):
#
#     mod = QuadraticProgram('particle_tracking')
#     vars = set()
#     linear = {}
#     quadratic = {}
#
#     for key, value in Q.items():
#         if key[0] == key[1]:
#             if key[0] not in vars:
#                 vars.add(key[0])
#                 mod.binary_var(name=key[0])
#                 linear.update({key[0]: value})
#
#         elif key[0] != key[1]:
#             if key[0] not in vars:
#                 vars.add(key[0])
#                 mod.binary_var(name=key[0])
#
#             if key[1] not in vars:
#                 vars.add(key[1])
#                 mod.binary_var(name=key[1])
#
#             quadratic.update({key: value})
#
#     mod.minimize(linear=linear, quadratic=quadratic)
#     return mod.to_ising()


# def Tracking_Hamiltonian_old(b_ij, a_i):
#
#     '''Given coupling strenghts b_ij and bias weights a_i return the tracking Hamiltonian as a Qiskit PauliOp object'''
#
#     nqubits = len(a_i)
#
#     H = I - I
#     H = H^nqubits
#
#     #prepare quadratic term
#     for i in range(nqubits):
#         for j in range(i):
#
#             n_left = nqubits - i - 1
#             n_middle = i - j - 1
#             n_right = j
#
#             temp = Z
#
#             if n_left > 0:
#
#                 id_left = I^n_left
#                 temp = id_left^temp
#
#             if n_middle > 0:
#
#                 id_middle = I^n_middle
#                 temp = temp^id_middle
#
#             temp = temp^Z
#
#             if n_right > 0:
#
#                 id_right = I^n_right
#                 temp = temp^id_right
#
#             H += b_ij[(i, j)] * temp
#
#     #prepare linear term
#     for i in range(nqubits):
#
#         bias = 0
#         for j in range(nqubits):
#             if j<i:
#                 bias += b_ij[(i,j)]
#             if i<j:
#                 bias += b_ij[(j,i)]
#
#         bias += 2*a_i[i]
#
#         n_left = nqubits - i - 1
#         n_right = i
#
#         temp = Z
#
#         if n_left > 0:
#
#             id_left = I^n_left
#             temp = id_left^temp
#
#         if n_right > 0:
#
#             id_right = I^n_right
#             temp = temp^id_right
#
#         H += -1.0 * bias * temp
#
#     return H


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


def solve_vqe_slices(Q_slices, **kwargs):

    def callback(*args):
    # check if interim results, since both interim results (list) and final results (dict) are returned
        if type(args[1]) is list:
            job_id, (nfev, parameters, energy, stddev) = args
            intermediate_info["nfev"].append(nfev)
            intermediate_info["parameters"].append(parameters)
            intermediate_info["energy"].append(energy)
            intermediate_info["stddev"].append(stddev)

    service = QiskitRuntimeService(channel='ibm_quantum')
    jobs_all_slices = []

    for slice in Q_slices:
        jobs_one_slice = []
        op, relations = tracking_hamiltonian(slice)
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

        for i in range(kwargs['vqe_repetitions']):
            intermediate_info = {"nfev": [], "parameters": [], "energy": [], "stddev": []}
            job = service.run(
                program_id='vqe',
                options=options,
                inputs=runtime_inputs,
                instance='ibm-q-desy/internal/tracking',
                callback=callback
            )
            jobs_one_slice.append({'job': job, 'intermediate_info': intermediate_info, 'relations': relations})

        jobs_all_slices.append(jobs_one_slice)

    results_all_slices = []

    for jobs_one_slice in jobs_all_slices:
        results_one_slice = []

        for job in jobs_one_slice:
            try:
                result = job['job'].result()
                intermediate_info = job['intermediate_info']
                relations = job['relations']
                result.update({'eigenstate_translated': translate_vqe_result(result, relations)})
                results_one_slice.append({'result': result, 'intermediate_info': intermediate_info, 'relations': relations})

            except:
                print('A single job failed')

        results_all_slices.append(results_one_slice)

    return results_all_slices


def solve_vqe_sub_qubos(Q, xplets, **kwargs):

    def callback(*args):
    # check if interim results, since both interim results (list) and final results (dict) are returned
        if type(args[1]) is list:
            job_id, (nfev, parameters, energy, stddev) = args
            intermediate_info["nfev"].append(nfev)
            intermediate_info["parameters"].append(parameters)
            intermediate_info["energy"].append(energy)
            intermediate_info["stddev"].append(stddev)

    service = QiskitRuntimeService(channel='ibm_quantum')
    op, relations = tracking_hamiltonian(Q)
    n_triplets = len(relations)
    state = ''.join(random.choices(['0', '1'], k = n_triplets))
    size = kwargs['sub_qubo_size']

    for i in range(kwargs['sub_qubo_iterations']):
        jobs_all_sub_qubos = []
        sub_qubos = create_sub_qubos(Q, op, relations, xplets, kwargs['sub_qubo_size'], state)
        n_slices = len(sub_qubos)

        for sub_qubo in sub_qubos:
            jobs_one_sub_qubo = []
            sub_op, sub_relations = tracking_hamiltonian(sub_qubo)
            n_qubits = len(sub_relations)
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

            for i in range(kwargs['vqe_repetitions']):
                intermediate_info = {"nfev": [], "parameters": [], "energy": [], "stddev": []}
                job = service.run(
                    program_id='vqe',
                    options=options,
                    inputs=runtime_inputs,
                    instance='ibm-q-desy/internal/tracking',
                    callback=callback
                )
                jobs_one_sub_qubo.append({'job': job, 'intermediate_info': intermediate_info, 'relations': sub_relations})

            jobs_all_sub_qubos.append(jobs_one_sub_qubo)

        results_all_sub_qubos = []
        state_list = list(state)

        for jobs_one_sub_qubo in jobs_all_sub_qubos:
            results_one_sub_qubo = []

            for job in jobs_one_sub_qubo:
                result = job['job'].result()
                intermediate_info = job['intermediate_info']
                sub_relations = job['relations']
                result.update({'eigenstate_translated': translate_vqe_result(result, sub_relations)})
                results_one_sub_qubo.append({'result': result, 'intermediate_info': intermediate_info, 'relations': sub_relations})

            results_all_sub_qubos.append(results_one_sub_qubo)
            result_lowest_energy = min(results_one_sub_qubo, key=lambda result: result['result']['optimal_value'])

            for triplet_id, value in result_lowest_energy['eigenstate_translated'].items():
                index = relations[triplet_id]
                state_list[index] = str(value)

        state = ''.join(state_list)

    return result_all_sub_qubos


def solve_eigensolver_sub_qubos(Q, xplets, **kwargs):

    H, relations = tracking_hamiltonian(Q)
    n_triplets = len(relations)
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


def solve_eigensolver_slices(Q_slices, vars_determined):

    n_slices = len(Q_slices)
    results = []

    for count, Q in enumerate(Q_slices):
        result_dict = {}
        op, relations = tracking_hamiltonian(Q)
        npme = NumPyMinimumEigensolver()
        result_eigensolver = npme.compute_minimum_eigenvalue(operator=op)
        counts_eigensolver = result_eigensolver.eigenstate.to_dict_fn().sample()
        result_dict.update({'eigenstate': counts_eigensolver})
        result_dict.update({'optimal_value': result_eigensolver.eigenvalue})
        result_translated = translate_vqe_result(result_dict, relations)
        result_dict.update({'eigenstate_translated': result_translated})
        result_dict.update({'vars_determined': vars_determined})
        results.append(result_dict)
        print('solved slice '+ str(count+1) + ' of ' + str(n_slices))

    return results
