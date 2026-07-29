"""
Task Suite Problem 7: 16-qubit measurement-feedback VQE.

The TensorCircuit-NG baseline uses cond_measure for ancilla measurements and
batches fixed trajectories with vmap for deterministic trajectory-averaged
energy optimization.
"""

import numpy as np
import optax

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("omeco-1-1")

PARAMS_PER_LAYER = 48


def initial_parameters(config):
    rng = np.random.default_rng(config["seed"])
    return K.convert_to_tensor(
        rng.normal(
            scale=config["initial_parameter_scale"],
            size=(config["n_layers"] * PARAMS_PER_LAYER,),
        ).astype(np.float32)
    )


def trajectory_status(config):
    rng = np.random.default_rng(config["seed"] + 1)
    return K.convert_to_tensor(
        rng.random(
            (config["n_trajectories"], config["n_layers"] * config["n_ancilla_qubits"]),
            dtype=np.float32,
        )
    )


def make_one_trajectory(config):
    n_data = config["n_data_qubits"]
    n_anc = config["n_ancilla_qubits"]
    n_qubits = config["n_qubits"]
    n_layers = config["n_layers"]
    transverse_field = config["transverse_field"]

    pauli_strings = []
    weights = []
    for i in range(n_data - 1):
        term = [0] * n_data
        term[i] = 3
        term[i + 1] = 3
        pauli_strings.append(term)
        weights.append(-1.0)
    for i in range(n_data):
        term = [0] * n_data
        term[i] = 1
        pauli_strings.append(term)
        weights.append(-transverse_field)
    hamiltonian = tc.quantum.PauliStringSum2COO(pauli_strings, weights)

    def measure_ancilla_register(c, status, offset):
        # One TensorCircuit state contraction contains the complete joint
        # ancilla distribution. Consume the same per-bit uniforms in the same
        # order as sequential cond_measure, then rebuild the normalized branch.
        state = K.reshape(c.state(), [2**n_data, 2**n_anc])
        probabilities = K.sum(K.abs(state) ** 2, axis=0)
        suffix = probabilities
        bits = []
        branch = K.cast(K.convert_to_tensor(0), "int32")
        for a in range(n_anc):
            suffix = K.reshape(suffix, [2, -1])
            p0 = K.sum(suffix[0]) / K.sum(suffix)
            bit = K.cast(status[offset + a] > p0, "int32")
            bits.append(bit)
            branch = 2 * branch + bit
            suffix = K.gather1d(suffix, K.reshape(bit, [1]))[0]

        column = K.gather1d(K.transpose(state), K.reshape(branch, [1]))[0]
        norm = K.sqrt(K.sum(K.abs(column) ** 2))
        ancilla = K.cast(K.onehot(branch, 2**n_anc), "complex64")
        collapsed = K.reshape(column[:, None] * ancilla[None, :], [-1])
        collapsed = collapsed / K.cast(norm + 1e-10, "complex64")
        return tc.Circuit(n_qubits, inputs=collapsed), bits

    def energy_of_data(c):
        # The final Z-measured ancillas remain computational-basis states
        # under diagonal RZZ feedback, so exactly one ancilla column is
        # nonzero. Contract the adaptive circuit once, recover the data state,
        # and evaluate all TFIM terms with one TensorCircuit-native operator.
        full_state = K.reshape(c.state(), [2**n_data, 2**n_anc])
        data_state = K.sum(full_state, axis=1)
        data_circuit = tc.Circuit(n_data, inputs=data_state)
        return tc.templates.measurements.operator_expectation(
            data_circuit, hamiltonian
        )

    def one_trajectory(params, status):
        c = tc.Circuit(n_qubits)
        pidx = 0
        sidx = 0

        for _ in range(n_layers):
            for q in range(n_data):
                c.ry(q, theta=params[pidx + q])
            pidx += n_data

            for a in range(n_anc):
                c.ry(n_data + a, theta=params[pidx + a])
            pidx += n_anc

            for a in range(n_anc):
                c.rzz(n_data + a, a, theta=params[pidx + a])
            pidx += n_anc

            for a in range(n_anc - 1):
                c.cnot(n_data + a, n_data + a + 1)

            theta0 = params[pidx : pidx + n_anc]
            pidx += n_anc
            theta1 = params[pidx : pidx + n_anc]
            pidx += n_anc

            c, bits = measure_ancilla_register(c, status, sidx)
            for a in range(n_anc):
                bit = bits[a]
                bitf = K.cast(bit, "float32")
                feedback_theta = theta0[a] + bitf * (theta1[a] - theta0[a])
                c.rz(
                    a,
                    theta=(1.0 - 2.0 * bitf) * feedback_theta,
                )
                sidx += 1

            for q in range(n_data - 1):
                c.cnot(q, q + 1)

            for q in range(n_data):
                c.rz(q, theta=params[pidx + q])
            pidx += n_data

        return energy_of_data(c)

    return one_trajectory


def run_solution(config):
    params = initial_parameters(config)
    status = trajectory_status(config)
    one_trajectory = make_one_trajectory(config)
    batched_trajectories = K.jit(K.vmap(one_trajectory, vectorized_argnums=1))
    optimizer = optax.adam(config["learning_rate"])

    def loss_fn(p):
        return K.mean(batched_trajectories(p, status))

    def train_step(p, state):
        value, grads = K.value_and_grad(loss_fn)(p)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, value

    train_step = K.jit(train_step)
    opt_state = optimizer.init(params)
    energy_history = []
    for _ in range(config["max_steps"]):
        params, opt_state, value = train_step(params, opt_state)
        energy_history.append(value)

    final_trajectory_energies = batched_trajectories(params, status)
    return {
        "energy_history": K.numpy(K.stack(energy_history)),
        "final_trajectory_energies": K.numpy(final_trajectory_energies),
    }
