import jax
import jax.numpy as jnp
import numpy as np

from tests.math_primitives import (
    calc_norm_vector_primitive,
    calc_norm_matrix_primitive,
    EPSILON,
)


def calc_phi_hatch_primitive(
    phi: jax.Array, n_t: jax.Array, vocab_size: int, n_topics: int
):
    phi_hatch = np.zeros_like(phi)
    for w in range(vocab_size):
        for t in range(n_topics):
            phi_hatch[w][t] = phi[w][t] * n_t[t]
        phi_hatch[w] = calc_norm_vector_primitive(phi_hatch[w])
    return jnp.array(phi_hatch)


def calc_attn_primitive(
    matrix: jax.Array,
    ctx_bounds: jax.Array,
    ctx_len: int,
    gamma: float,
):
    n_words, n_topics = matrix.shape

    attn = []
    for w in range(n_words):
        prefix_context_vec, prefix_context_weights = [], []
        for i in range(1, ctx_len + 1):
            if w - i >= 0 and not ctx_bounds[w - i + 1]:
                prefix_context_vec.append(matrix[w - i])
                prefix_context_weights.append(gamma * (1 - gamma) ** i)
            else:
                break

        suffix_context_vec, suffix_context_weights = [], []
        for i in range(1, ctx_len + 1):
            if w + i < n_words and not ctx_bounds[w + i]:
                suffix_context_vec.append(matrix[w + i])
                suffix_context_weights.append(gamma * (1 - gamma) ** i)
            else:
                break

        context_weights = jnp.array(
            prefix_context_weights[::-1] + suffix_context_weights
        )
        context_weights = calc_norm_vector_primitive(context_weights)

        context_vec = jnp.array(prefix_context_vec[::-1] + suffix_context_vec)
        context_vec = context_vec * context_weights[:, None]
        context_vec = jnp.sum(context_vec, axis=0)

        if context_vec.shape == (0,):
            # no words in context
            context_vec = jnp.zeros(shape=(n_topics,))

        assert context_vec.shape == (n_topics,)

        attn.append(context_vec)
    return jnp.array(attn)


def calc_attn_transposed_primitive(
    matrix: jax.Array,
    ctx_bounds: jax.Array,
    ctx_len: int,
    gamma: float,
):
    n_words, n_topics = matrix.shape

    transposed_attn = np.zeros((n_words, n_topics), dtype=matrix.dtype)
    for w in range(n_words):
        prefix_positions, prefix_weights = [], []
        for i in range(1, ctx_len + 1):
            if w - i >= 0 and not ctx_bounds[w - i + 1]:
                prefix_positions.append(w - i)
                prefix_weights.append(gamma * (1 - gamma) ** i)
            else:
                break

        suffix_positions, suffix_weights = [], []
        for i in range(1, ctx_len + 1):
            if w + i < n_words and not ctx_bounds[w + i]:
                suffix_positions.append(w + i)
                suffix_weights.append(gamma * (1 - gamma) ** i)
            else:
                break

        context_positions = prefix_positions[::-1] + suffix_positions
        context_weights = np.array(prefix_weights[::-1] + suffix_weights)

        if context_weights.shape[0] == 0:
            continue

        context_weights = calc_norm_vector_primitive(context_weights)

        for pos, weight in zip(context_positions, context_weights):
            for t in range(n_topics):
                transposed_attn[pos][t] += weight * matrix[w][t]

    return transposed_attn


def calc_theta_primitive(
    data: jax.Array,
    phi_hatch: jax.Array,
    doc_bounds: jax.Array,
    ctx_len: int,
    gamma: float,
):
    phi_it_hatch = phi_hatch[data]
    theta = calc_attn_primitive(
        matrix=phi_it_hatch, ctx_bounds=doc_bounds, ctx_len=ctx_len, gamma=gamma
    )
    return theta


def calc_p_it_primitive(
    data: jax.Array,
    phi: jax.Array,
    theta: jax.Array,
    n_topics: int,
    n_words: int,
):
    p_it = np.zeros((n_words, n_topics))
    for w in range(n_words):
        for t in range(n_topics):
            word = data[w]
            p_it[w][t] = phi[word][t] * theta[w][t]
        p_it[w] = calc_norm_vector_primitive(p_it[w])
    return jnp.array(p_it)


def calc_p_ti_primitive(
    phi: jax.Array,
    theta: jax.Array,
    n_t: jax.Array,
    n_topics: int,
    n_words: int,
):
    assert phi.shape == (n_words, n_topics)
    assert theta.shape == (n_words, n_topics)

    p_ti = np.zeros((n_topics, n_words))
    for w in range(n_words):
        for t in range(n_topics):
            p_ti[t][w] = phi[w][t] * theta[w][t] / (n_t[t] + EPSILON)
        p_ti[:, w] = calc_norm_vector_primitive(p_ti[:, w])
    return jnp.array(p_ti).T


def calc_n_t_primitive(p_ti: jax.Array, n_topics: int, n_words: int):
    assert p_ti.shape == (n_words, n_topics)

    n_t = np.zeros((n_topics,))
    for w in range(n_words):
        for t in range(n_topics):
            n_t[t] += p_ti[w][t]
    return jnp.array(n_t)


def calc_phi_wt_primitive(
    data: jax.Array,
    p_ti: jax.Array,
    vocab_size: int,
    n_topics: int,
    n_words: int,
):
    assert p_ti.shape == (n_words, n_topics)

    # not testing partial derivatives here, trusting in jax.grad
    phi = np.zeros((vocab_size, n_topics))
    for i in range(n_words):
        for t in range(n_topics):
            word_token = data[i]
            phi[word_token][t] += p_ti[i][t]
    phi = calc_norm_matrix_primitive(phi)
    return phi


def calc_N_tw_primitive(
    data: jax.Array,
    p_ti: jax.Array,
    theta: jax.Array,
    doc_bounds: jax.Array,
    vocab_size: int,
    n_topics: int,
    n_words: int,
    ctx_len: int,
    gamma: float,
):
    assert p_ti.shape == (n_words, n_topics)
    assert theta.shape == (n_words, n_topics)

    # create one-hot [w_i = w]
    q_iw = np.zeros((n_words, vocab_size))
    for i in range(n_words):
        word_token = data[i]
        q_iw[i][word_token] = 1
    q_iw = calc_attn_primitive(
        matrix=q_iw, ctx_bounds=doc_bounds, ctx_len=ctx_len, gamma=gamma
    )

    N_tw = np.zeros((n_topics, vocab_size))
    for t in range(n_topics):
        for w in range(vocab_size):
            for i in range(n_words):
                N_tw[t][w] += q_iw[i][w] * p_ti[i][t] / (theta[i][t] + EPSILON)
    return N_tw


def calc_phi_tw_primitive(
    data: jax.Array,
    p_ti: jax.Array,
    N_tw: jax.Array,
    phi_old: jax.Array,
    vocab_size: int,
    n_topics: int,
    n_words: int,
):
    assert p_ti.shape == (n_words, n_topics)
    assert N_tw.shape == (n_topics, vocab_size)
    assert phi_old.shape == (vocab_size, n_topics)

    # not testing partial derivatives here, trusting in jax.grad
    n_tw = np.zeros((n_topics, vocab_size))
    n_w = np.zeros(vocab_size)
    for i in range(n_words):
        for t in range(n_topics):
            word_token = data[i]
            n_tw[t][word_token] += p_ti[i][t]
            n_w[word_token] += p_ti[i][t]

    coeff = n_tw / (n_w + EPSILON)[None, :]
    phi = n_tw + coeff * N_tw
    phi = calc_norm_matrix_primitive(phi).T
    return phi
