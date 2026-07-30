"""
Challenge Suite Problem 8: 7x7 mixed-axis grid sampling.

The solution builds the TensorCircuit tensor network directly and samples from it
without constructing the full 2^49 statevector.
"""

import numpy as np

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")
import omeco

tc.set_contractor("custom", optimizer=omeco.TreeSA(ntrials=4, niters=4), preprocessing=True)


def ry_angle(row, col, config):
    return (
        config["ry_offset"]
        + config["ry_row_sin_scale"]
        * np.sin(config["ry_row_sin_frequency"] * (row + 1))
        + config["ry_col_cos_scale"]
        * np.cos(config["ry_col_cos_frequency"] * (col + 1))
        + config["ry_diag_sin_scale"]
        * np.sin(config["ry_diag_sin_frequency"] * (row + col + 2))
    )


def rzz_angle(row, col, edge_index, config):
    return (
        config["rzz_offset"]
        + config["rzz_edge_sin_scale"]
        * np.sin(config["rzz_edge_sin_frequency"] * (edge_index + 1))
        + config["rzz_site_cos_scale"]
        * np.cos(config["rzz_site_cos_frequency"] * (2 * row + col + 1))
    )


def rxx_angle(row, col, edge_index, config):
    return (
        config["rxx_offset"]
        + config["rxx_edge_cos_scale"]
        * np.cos(config["rxx_edge_cos_frequency"] * (edge_index + 1))
        + config["rxx_site_sin_scale"]
        * np.sin(config["rxx_site_sin_frequency"] * (row + 2 * col + 1))
    )


def rx_angle(row, col, config):
    return (
        config["rx_offset"]
        + config["rx_row_cos_scale"]
        * np.cos(config["rx_row_cos_frequency"] * (row + 1))
        - config["rx_col_sin_scale"]
        * np.sin(config["rx_col_sin_frequency"] * (col + 1))
        + config["rx_diag_cos_scale"]
        * np.cos(config["rx_diag_cos_frequency"] * (row + col + 2))
    )


def build_circuit(config):
    n_side = config["grid_side"]
    circuit = tc.Circuit(config["n_qubits"])

    for row in range(n_side):
        for col in range(n_side):
            qubit = (row * n_side + col + 1) % config["n_qubits"]
            circuit.ry(qubit, theta=ry_angle(row, col, config))

    edge_index = 0
    for row in range(n_side):
        for col in range(n_side - 1):
            left = (row * n_side + col + 1) % config["n_qubits"]
            right = left + 1
            circuit.rzz(left, right, theta=rzz_angle(row, col, edge_index, config))
            edge_index += 1

    edge_index = 0
    for row in range(n_side - 1):
        for col in range(n_side):
            left = (row * n_side + col + 1) % config["n_qubits"]
            right = (row + 1) * n_side + col
            circuit.rxx(left, right, theta=rxx_angle(row, col, edge_index, config))
            edge_index += 1

    for row in range(n_side):
        for col in range(n_side):
            qubit = (row * n_side + col + 1) % config["n_qubits"]
            circuit.rx(qubit, theta=rx_angle(row, col, config))

    return circuit


def run_solution(config):
    circuit = build_circuit(config)

    rng = np.random.default_rng(2033)
    status = K.convert_to_tensor(
        rng.random((config["n_samples"], config["n_qubits"]), dtype=np.float32)
    )

    def sample_one(seed):
        sample, _ = circuit.perfect_sampling(seed)
        return sample

    sample_batch = K.jit(K.vmap(sample_one))
    sampling_batch_size = 256
    sample_chunks = []
    for start in range(0, config["n_samples"], sampling_batch_size):
        stop = start + sampling_batch_size
        sample_chunks.append(K.numpy(sample_batch(status[start:stop])))
    samples = np.concatenate(sample_chunks, axis=0)
    return {"samples": np.asarray(samples, dtype=np.int32)}
