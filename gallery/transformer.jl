# dense_linear(x, w, b, y, n, d_in, d_out, w_off, b_off)
#
# Dense layer applied to every row of x: y = x * W + b, with W and b
# read from a shared, per-layer-stacked weight/bias array at an offset.
#
# x: input array, length n * d_in, row-major (row i, column c at
#    (i-1)*d_in+c)
# w: weight array; this call's (d_in, d_out) block starts at w_off
# b: bias array; this call's length-d_out block starts at b_off
# y: output array, length n * d_out, filled in place
# n: number of rows
# d_in: input width
# d_out: output width
# w_off: offset into w where this call's weight block starts
# b_off: offset into b where this call's bias block starts
function dense_linear(x, w, b, y, n, d_in, d_out, w_off, b_off)
    for idx = 1:n * d_out
        i = div(idx - 1, d_out) + 1
        j = mod(idx - 1, d_out) + 1
        s = 0.0
        # contraction over the input width is a genuine accumulation
        for i_p = 1:d_in
            s = s + x[(i - 1) * d_in + i_p] * w[w_off + (i_p - 1) * d_out + j]
        end
        y[(i - 1) * d_out + j] = s + b[b_off + j]
    end
    return nothing
end

# relu(buf, n)
#
# Rectified linear unit, applied in place: every negative entry becomes
# zero, every non-negative entry is unchanged.
#
# buf: array to rectify, length at least n
# n: number of elements to rectify
function relu(buf, n)
    for i = 1:n
        buf[i] = max(buf[i], 0.0)
    end
    return nothing
end

# dense_relu(x, w, b, y, n, d_in, d_out, w_off, b_off)
#
# Dense layer followed by a ReLU, applied to every row of x. Same
# argument meaning as dense_linear.
#
# x: input array, length n * d_in, row-major
# w: weight array; this call's (d_in, d_out) block starts at w_off
# b: bias array; this call's length-d_out block starts at b_off
# y: output array, length n * d_out, filled in place
# n: number of rows
# d_in: input width
# d_out: output width
# w_off: offset into w where this call's weight block starts
# b_off: offset into b where this call's bias block starts
function dense_relu(x, w, b, y, n, d_in, d_out, w_off, b_off)
    n_out = n * d_out
    dense_linear(x, w, b, y, n, d_in, d_out, w_off, b_off)
    relu(y, n_out)
    return nothing
end

# residual_add(a, b, y, n)
#
# Elementwise sum of two arrays: y = a + b.
#
# a: first input array, length at least n
# b: second input array, length at least n
# y: output array, length at least n, filled in place
# n: number of elements to add
function residual_add(a, b, y, n)
    for idx = 1:n
        y[idx] = a[idx] + b[idx]
    end
    return nothing
end

# layernorm(x, gain, bias, y, n, d, off, eps)
#
# Row-wise layer normalization: each row of x is centered, scaled to
# unit variance, then rescaled and shifted by gain/bias. Every row is
# normalized independently of the others.
#
# x: input array, length n * d, row-major
# gain, bias: scale and shift parameters; this call's length-d block
#             starts at off in each
# y: output array, length n * d, filled in place
# n: number of rows
# d: row width
# off: offset into gain/bias where this call's block starts
# eps: variance floor added before the square root
function layernorm(x, gain, bias, y, n, d, off, eps)
    for i = 1:n
        s = 0.0
        # mean requires a running sum, so this inner loop is sequential
        for i_j = 1:d
            s = s + x[(i - 1) * d + i_j]
        end
        row_mean = s / d
        s2 = 0.0
        # variance likewise accumulates sequentially
        for i_j = 1:d
            diff = x[(i - 1) * d + i_j] - row_mean
            s2 = s2 + diff * diff
        end
        row_var = s2 / d
        denom = sqrt(row_var + eps)
        # the final rescale is independent across columns
        for j = 1:d
            kk = (i - 1) * d + j
            y[kk] = (x[kk] - row_mean) / denom * gain[off + j] + bias[off + j]
        end
    end
    return nothing
end

# copy_array(src, dst, n)
#
# Copies the first n elements of src into dst.
#
# src: input array, length at least n
# dst: output array, length at least n, filled in place
# n: number of elements to copy
function copy_array(src, dst, n)
    for idx = 1:n
        dst[idx] = src[idx]
    end
    return nothing
end

# attention_head(q, k, v, scores, probs, ctx, head_offset, score_off,
#                n, dk, d, inv_sqrt_dk)
#
# Scaled dot-product attention for one head: raw scores from q and k,
# a row-wise softmax into probs, then the attended sum of v.
#
# q, k, v: query/key/value arrays, length n * d, row-major; this head's
#          own dk columns start at column head_offset in every row
# scores, probs: scratch arrays, length (number of heads) * n * n --
#                this head's own private n*n slice starts at score_off
# ctx: output array, length n * d, row-major; this head's own dk output
#      columns start at column head_offset in every row
# head_offset: column offset of this head's slice within q, k, v, ctx
# score_off: offset of this head's slice within scores/probs
# n: number of tokens
# dk: this head's width
# d: full model width (number of columns in q, k, v, ctx)
# inv_sqrt_dk: 1/sqrt(dk), the attention scaling factor
function attention_head(q, k, v, scores, probs, ctx, head_offset, score_off, n, dk, d, inv_sqrt_dk)
    # scores = (q_head * k_head^T) / sqrt(dk); every (i, j) pair is
    # independent, and dk is fixed
    for idx2 = 1:n * n
        i = div(idx2 - 1, n) + 1
        j = mod(idx2 - 1, n) + 1
        s = 0.0
        # the dot product over the head's dk dims is a genuine accumulation
        for i_p = 1:dk
            s = s + q[(i - 1) * d + head_offset + i_p] * k[(j - 1) * d + head_offset + i_p]
        end
        scores[score_off + (i - 1) * n + j] = s * inv_sqrt_dk
    end

    # row-wise softmax of scores into probs; each row is independent
    for i = 1:n
        row_max = scores[score_off + (i - 1) * n + 1]
        # running max requires a sequential scan; max() avoids a branch
        for i_j = 2:n
            row_max = max(row_max, scores[score_off + (i - 1) * n + i_j])
        end
        # shifted exponentials are independent across columns
        for j = 1:n
            kk = score_off + (i - 1) * n + j
            probs[kk] = exp(scores[kk] - row_max)
        end
        row_sum = 0.0
        # normalizing sum requires a running total, so it stays sequential
        for i_j = 1:n
            row_sum = row_sum + probs[score_off + (i - 1) * n + i_j]
        end
        # final division is independent across columns
        for j = 1:n
            kk = score_off + (i - 1) * n + j
            probs[kk] = probs[kk] / row_sum
        end
    end

    # ctx head slice = probs * v_head; every (token, head-dim) output
    # entry is independent, and dk is fixed
    for idx3 = 1:n * dk
        i = div(idx3 - 1, dk) + 1
        p = mod(idx3 - 1, dk) + 1
        s = 0.0
        # summing over the attended tokens is a genuine accumulation
        for i_j = 1:n
            s = s + probs[score_off + (i - 1) * n + i_j] * v[(i_j - 1) * d + head_offset + p]
        end
        ctx[(i - 1) * d + head_offset + p] = s
    end
    return nothing
end

# multihead_attention(q, k, v, scores, probs, ctx, n, dk, h, d, inv_sqrt_dk)
#
# Runs every attention head. Each head is computed independently of the
# others, but scores/probs need their own private n*n slice per head --
# a single shared n*n buffer, fully overwritten every call, only avoids
# a cross-call dependency under strictly sequential (CPU) execution.
#
# q, k, v: query/key/value arrays, length n * d, row-major
# scores, probs: scratch arrays, length h * n * n
# ctx: output array, length n * d, row-major, filled in place
# n: number of tokens
# dk: per-head width
# h: number of attention heads
# d: full model width, h*dk, passed in rather than recomputed so it stays
#    a loop-invariant value in every caller up to the encoder-layer loop
# inv_sqrt_dk: 1/sqrt(dk), the attention scaling factor
function multihead_attention(q, k, v, scores, probs, ctx, n, dk, h, d, inv_sqrt_dk)
    for hh = 1:h
        head_offset = (hh - 1) * dk
        score_off = (hh - 1) * n * n
        attention_head(q, k, v, scores, probs, ctx, head_offset, score_off, n, dk, d, inv_sqrt_dk)
    end
    return nothing
end

# transformer_layer(x, wq, bq, wk, bk, wv, bv, wo, bo, ln1_gain, ln1_bias,
#                    w1, b1, w2, b2, ln2_gain, ln2_bias,
#                    q, k, v, scores, probs, ctx, attn_out, resid1, normed1,
#                    ff_hidden, ff_out, resid2, x_next,
#                    n, dk, h, dff, eps, d, n_d, inv_sqrt_dk,
#                    w_off, b_off, ln_off, w1_off, b1_off, w2_off, b2_off, ln2_off)
#
# Runs one Transformer encoder layer in place on x: self-attention with
# its output projection, a residual add and layer norm, a position-wise
# feed-forward block, and a second residual add and layer norm. Every
# per-layer weight/bias is read from its own offset into a shared,
# stacked-over-layers array, so the same arrays serve every layer.
#
# x: input array, length n*d, row-major; overwritten in place with this
#    layer's output
# wq, bq, wk, bk, wv, bv, wo, bo: self-attention parameters, this layer's
#    block starting at w_off (weights) / b_off (biases)
# ln1_gain, ln1_bias: first layer-norm parameters, this layer's block
#    starting at ln_off
# w1, b1: feed-forward hidden-layer parameters, this layer's block
#    starting at w1_off / b1_off
# w2, b2: feed-forward output-layer parameters, this layer's block
#    starting at w2_off / b2_off
# ln2_gain, ln2_bias: second layer-norm parameters, this layer's block
#    starting at ln2_off
# q, k, v: scratch arrays of length n*d for the projected query/key/value
# scores, probs: scratch arrays of length h*n*n for the attention scores
#    and weights
# ctx: scratch array of length n*d for the concatenated head outputs
# attn_out: scratch array of length n*d for the attention output projection
# resid1, normed1: scratch arrays of length n*d for the first residual block
# ff_hidden: scratch array of length n*dff for the feed-forward hidden layer
# ff_out: scratch array of length n*d for the feed-forward output
# resid2: scratch array of length n*d for the second residual sum
# x_next: scratch array of length n*d holding this layer's freshly
#    computed output before it is copied back into x
# n: number of tokens in the sequence
# dk: per-head width
# h: number of attention heads
# dff: feed-forward hidden width
# eps: layer-norm epsilon
# d, n_d, inv_sqrt_dk: the model width h*dk, n*d, and 1/sqrt(dk) --
#    every one of these is the same value on every call, so the caller
#    computes each once and passes it down here rather than every layer
#    recomputing it fresh
# w_off, b_off: offset of this layer's self-attention weight/bias block
# ln_off: offset of this layer's first layer-norm block
# w1_off, b1_off: offset of this layer's feed-forward hidden-layer block
# w2_off, b2_off: offset of this layer's feed-forward output-layer block
# ln2_off: offset of this layer's second layer-norm block
function transformer_layer(x, wq, bq, wk, bk, wv, bv, wo, bo, ln1_gain, ln1_bias, w1, b1, w2, b2, ln2_gain, ln2_bias, q, k, v, scores, probs, ctx, attn_out, resid1, normed1, ff_hidden, ff_out, resid2, x_next, n, dk, h, dff, eps, d, n_d, inv_sqrt_dk, w_off, b_off, ln_off, w1_off, b1_off, w2_off, b2_off, ln2_off)
    # --- query/key/value projections ---
    dense_linear(x, wq, bq, q, n, d, d, w_off, b_off)
    dense_linear(x, wk, bk, k, n, d, d, w_off, b_off)
    dense_linear(x, wv, bv, v, n, d, d, w_off, b_off)

    # --- multi-head scaled dot-product attention ---
    multihead_attention(q, k, v, scores, probs, ctx, n, dk, h, d, inv_sqrt_dk)

    # --- output projection, then the first residual add and layer norm ---
    dense_linear(ctx, wo, bo, attn_out, n, d, d, w_off, b_off)
    residual_add(x, attn_out, resid1, n_d)
    layernorm(resid1, ln1_gain, ln1_bias, normed1, n, d, ln_off, eps)

    # --- position-wise feed-forward block ---
    dense_relu(normed1, w1, b1, ff_hidden, n, d, dff, w1_off, b1_off)
    dense_linear(ff_hidden, w2, b2, ff_out, n, dff, d, w2_off, b2_off)

    # --- second residual add and layer norm ---
    residual_add(normed1, ff_out, resid2, n_d)
    layernorm(resid2, ln2_gain, ln2_bias, x_next, n, d, ln2_off, eps)

    # carry this layer's output into x as the next layer's input
    copy_array(x_next, x, n_d)
    return nothing
end

# transformer(x, wq, bq, wk, bk, wv, bv, wo, bo, ln1_gain, ln1_bias,
#                    w1, b1, w2, b2, ln2_gain, ln2_bias,
#                    q, k, v, scores, probs, ctx, attn_out, resid1, normed1,
#                    ff_hidden, ff_out, resid2, x_next,
#                    n, dk, h, dff, n_layers, eps)
#
# A Transformer encoder stack: query/key/value projection, multi-head
# scaled dot-product attention, output projection, both residual-add-
# and-layer-norm steps, and the position-wise feed-forward block, in
# n_layers stacked layers. Pre-processing (token embedding, positional
# encoding) and post-processing (an output head) are outside this
# kernel's scope; x is taken to already be an embedded token sequence.
#
# Per-layer weights are passed as single flat arrays stacked over
# n_layers (for example wq has length n_layers*d*d; layer l's block
# starts at flat offset (l-1)*d*d), since a kernel takes no structs.
#
# x: input/output array of length n*d, row-major (token t, dim c at
#    (t-1)*d+c), overwritten in place with the encoder-stack output
# wq, bq, wk, bk, wv, bv, wo, bo: self-attention parameters stacked over
#    n_layers (wq/wk/wv/wo length n_layers*d*d, bq/bk/bv/bo length n_layers*d)
# ln1_gain, ln1_bias: first layer-norm parameters, length n_layers*d
# w1, b1, w2, b2: feed-forward parameters (w1 length n_layers*d*dff, b1
#    length n_layers*dff, w2 length n_layers*dff*d, b2 length n_layers*d)
# ln2_gain, ln2_bias: second layer-norm parameters, length n_layers*d
# q, k, v: scratch arrays of length n*d for the projected query/key/value
# scores, probs: scratch arrays of length h*n*n
# ctx: scratch array of length n*d for the concatenated head outputs
# attn_out: scratch array of length n*d for the attention output projection
# resid1, normed1: scratch arrays of length n*d for the first residual block
# ff_hidden: scratch array of length n*dff for the feed-forward hidden layer
# ff_out: scratch array of length n*d for the feed-forward output
# resid2: scratch array of length n*d for the second residual sum
# x_next: scratch array of length n*d holding each layer's freshly
#    computed output before it is copied back into x
# n: number of tokens in the sequence
# dk: per-head width
# h: number of attention heads
# dff: feed-forward hidden width
# n_layers: number of stacked encoder layers
# eps: layer-norm epsilon
function transformer(x, wq, bq, wk, bk, wv, bv, wo, bo, ln1_gain, ln1_bias, w1, b1, w2, b2, ln2_gain, ln2_bias, q, k, v, scores, probs, ctx, attn_out, resid1, normed1, ff_hidden, ff_out, resid2, x_next, n, dk, h, dff, n_layers, eps)
    # the model width and the attention scale are the same on every
    # layer, so compute each once here rather than once per layer
    d = h * dk
    n_d = n * d
    inv_sqrt_dk = 1.0 / sqrt(dk)

    # layers form a genuine sequential chain: layer l+1 consumes layer
    # l's output, carried through x
    for i_l = 1:n_layers
        w_offset = (i_l - 1) * d * d
        b_offset = (i_l - 1) * d
        ln_offset = (i_l - 1) * d
        w1_offset = (i_l - 1) * d * dff
        b1_offset = (i_l - 1) * dff
        w2_offset = (i_l - 1) * dff * d
        b2_offset = (i_l - 1) * d
        ln2_offset = (i_l - 1) * d
        transformer_layer(x, wq, bq, wk, bk, wv, bv, wo, bo, ln1_gain, ln1_bias, w1, b1, w2, b2, ln2_gain, ln2_bias, q, k, v, scores, probs, ctx, attn_out, resid1, normed1, ff_hidden, ff_out, resid2, x_next, n, dk, h, dff, eps, d, n_d, inv_sqrt_dk, w_offset, b_offset, ln_offset, w1_offset, b1_offset, w2_offset, b2_offset, ln2_offset)
    end

    return nothing
end