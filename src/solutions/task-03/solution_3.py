"""Task 3: exact product-state contraction of the post-selected circuit."""

import numpy as np
import optax
import tensorcircuit as tc


K = tc.set_backend("jax")
tc.set_dtype("complex64")


def initial_parameters(config):
    rng = np.random.default_rng(2027)
    repeats = config["n_steps"] // 2
    n = config["n_qubits"]

    def normal(shape):
        return K.convert_to_tensor(
            rng.normal(scale=0.02, size=shape).astype(np.float32)
        )

    return {
        "even": {
            "rx": normal((repeats, n)),
            "xx": normal((repeats, n // 2)),
            "zz": normal((repeats, n // 2)),
        },
        "odd": {
            "rx": normal((repeats, n)),
            "xx": normal((repeats, n // 2 - 1)),
            "zz": normal((repeats, n // 2 - 1)),
        },
    }


def pair_map(odd, even, xx, zz, odd_rx, even_rx, even_left):
    rxx = K.reshape(tc.gates.rxx_gate(theta=xx).tensor, [4, 4])
    rzz = K.reshape(tc.gates.rzz_gate(theta=zz).tensor, [4, 4])
    ro = tc.gates.rx_gate(theta=odd_rx).tensor
    re = tc.gates.rx_gate(theta=even_rx).tensor
    if even_left:
        state = K.kron(even, odd)
        rotations = K.kron(re, ro)
    else:
        state = K.kron(odd, even)
        rotations = K.kron(ro, re)
    state = K.tensordot(
        rotations,
        K.tensordot(rzz, K.tensordot(rxx, state, 1), 1),
        1,
    )
    state = K.reshape(state, [2, 2])
    selected = state[0, :] if even_left else state[:, 0]
    probability = K.real(K.sum(K.conj(selected) * selected))
    return selected / K.sqrt(probability + 1e-12), probability


def cooling_product(params, config):
    n_survivors = config["n_qubits"] // 2
    zero = K.convert_to_tensor(np.array([1.0, 0.0], dtype=np.complex64))
    plus = tc.gates.h().tensor[:, 0]
    states = K.stack([plus for _ in range(n_survivors)])
    even_inputs = K.stack(
        [plus] + [zero for _ in range(config["n_steps"] // 2 - 1)]
    )

    def block_step(states, inputs):
        even_params, odd_params, even_input = inputs
        log_probabilities = []
        next_states = []
        for k, odd in enumerate(states):
            odd, probability = pair_map(
                odd,
                even_input,
                even_params["xx"][k],
                even_params["zz"][k],
                even_params["rx"][2 * k + 1],
                even_params["rx"][2 * k],
                True,
            )
            next_states.append(odd)
            log_probabilities.append(K.log(probability + 1e-12))
        states = next_states

        even_only = K.tensordot(
            tc.gates.rx_gate(theta=odd_params["rx"][0]).tensor, zero, 1
        )
        probability = K.real(K.conj(even_only[0]) * even_only[0])
        log_probabilities.append(K.log(probability + 1e-12))
        next_states = []
        for k, odd in enumerate(states[:-1]):
            odd, probability = pair_map(
                odd,
                zero,
                odd_params["xx"][k],
                odd_params["zz"][k],
                odd_params["rx"][2 * k + 1],
                odd_params["rx"][2 * k + 2],
                False,
            )
            next_states.append(odd)
            log_probabilities.append(K.log(probability + 1e-12))
        next_states.append(
            K.tensordot(
                tc.gates.rx_gate(theta=odd_params["rx"][-1]).tensor,
                states[-1],
                1,
            )
        )
        return K.stack(next_states), K.stack(log_probabilities)

    states, log_probabilities = K.jaxy_scan(
        block_step,
        states,
        (params["even"], params["odd"], even_inputs),
    )
    return states, K.reshape(log_probabilities, [-1])


def one_qubit_expectation(state, operator):
    return K.real(
        K.sum(K.conj(state) * K.tensordot(operator.tensor, state, 1))
    )


def observables(params, config):
    states, log_probabilities = cooling_product(params, config)
    xs = K.stack([one_qubit_expectation(state, tc.gates.x()) for state in states])
    zs = K.stack([one_qubit_expectation(state, tc.gates.z()) for state in states])
    zz_sum = 2.0 * K.sum(zs[:-1]) + zs[-1]
    energy = -(zz_sum + config["transverse_field"] * K.sum(xs))
    energy_density = energy / config["n_qubits"]
    mean_log_probability = K.mean(log_probabilities)
    success_probability = K.exp(K.sum(log_probabilities))
    loss = (
        energy_density
        - config["log_probability_weight"] * mean_log_probability
    )
    return loss, (energy_density, success_probability, mean_log_probability)


def run_solution(config):
    params = initial_parameters(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def train_step(p, state):
        (loss, aux), grads = K.value_and_grad(observables, has_aux=True)(p, config)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, loss, aux

    train_step = K.jit(train_step)
    rows = []
    for _ in range(config["max_steps"]):
        params, opt_state, loss, aux = train_step(params, opt_state)
        energy, success, mean_log = aux
        rows.append(K.stack([energy, success, mean_log, loss]))
    history = K.numpy(K.stack(rows))
    return {
        "energy_density_history": history[:, 0],
        "success_probability_history": history[:, 1],
        "mean_log_probability_history": history[:, 2],
        "loss_history": history[:, 3],
    }
