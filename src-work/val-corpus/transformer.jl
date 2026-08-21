# transformer.jl
#
# A Transformer encoder stack as a single skill-jade-compliant kernel. All
# work (query/key/value projection, multi-head scaled dot-product
# attention, output projection, both residual-add-and-layer-norm steps,
# and the position-wise feed-forward block) is inlined into one function
# body; there are no helper functions to call. Every array is caller
# allocated and passed in; the kernel allocates nothing and returns
# nothing, writing its result into x in place.
#
# Pre-processing (token embedding, positional encoding) and post-
# processing (output head) are disregarded, per the request; x is taken
# to already be an embedded token sequence.
#
# Per-layer weights are passed as single flat arrays stacked over
# n_layers (e.g. wq has length n_layers*d*d; layer l's block starts at
# flat offset (l-1)*d*d), since a single kernel takes no structs.
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
# scores, probs: scratch arrays of length h*n*n -- one private n*n slice
#    per head for that head's raw scores and attention weights
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
#    (the model width d is derived in the body as h*dk, not passed)
# dff: feed-forward hidden width
# n_layers: number of stacked encoder layers
# eps: layer-norm epsilon
function transformer(x, wq, bq, wk, bk, wv, bv, wo, bo, ln1_gain, ln1_bias, w1, b1, w2, b2, ln2_gain, ln2_bias, q, k, v, scores, probs, ctx, attn_out, resid1, normed1, ff_hidden, ff_out, resid2, x_next, n, dk, h, dff, n_layers, eps)
    d = h * dk
    inv_sqrt_dk = 1.0 / sqrt(dk)
    n_d = n * d
    n_dff = n * dff

    # layers form a genuine sequential chain: layer l+1 consumes layer l's output
    for i_seq_l = 1:n_layers
        w_offset = (i_seq_l - 1) * d * d
        b_offset = (i_seq_l - 1) * d
        ln_offset = (i_seq_l - 1) * d
        w1_offset = (i_seq_l - 1) * d * dff
        b1_offset = (i_seq_l - 1) * dff
        w2_offset = (i_seq_l - 1) * dff * d
        b2_offset = (i_seq_l - 1) * d
        ln2_offset = (i_seq_l - 1) * d

        # --- query projection: q = x * Wq + bq ---
        # every (token, out-dim) entry is independent and d is fixed, so fuse
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s = 0.0
            # contraction over the model width is a genuine accumulation
            for i_seq_p = 1:d
                s = s + x[(i - 1) * d + i_seq_p] * wq[w_offset + (i_seq_p - 1) * d + j]
            end
            q[(i - 1) * d + j] = s + bq[b_offset + j]
        end

        # --- key projection: k = x * Wk + bk ---
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s = 0.0
            for i_seq_p = 1:d
                s = s + x[(i - 1) * d + i_seq_p] * wk[w_offset + (i_seq_p - 1) * d + j]
            end
            k[(i - 1) * d + j] = s + bk[b_offset + j]
        end

        # --- value projection: v = x * Wv + bv ---
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s = 0.0
            for i_seq_p = 1:d
                s = s + x[(i - 1) * d + i_seq_p] * wv[w_offset + (i_seq_p - 1) * d + j]
            end
            v[(i - 1) * d + j] = s + bv[b_offset + j]
        end

        # --- multi-head scaled dot-product attention ---
        # each head is computed independently of the others; scores/probs
        # need their own private n*n slice per head, though -- a single
        # shared n*n buffer "fully overwritten each iteration" only avoids
        # a cross-iteration dependency under strictly sequential (CPU)
        # execution; once this loop is GPU-split (one thread per head),
        # concurrent heads would race on the same shared slice
        for hh = 1:h
            head_offset = (hh - 1) * dk
            score_off = (hh - 1) * n * n

            # scores = (q_head * k_head^T) / sqrt(dk)
            # every (i, j) score pair is independent, and dk is fixed
            for idx2 = 1:n * n
                i = div(idx2 - 1, n) + 1
                j = mod(idx2 - 1, n) + 1
                s = 0.0
                # the dot product over the head's dk dims is a genuine accumulation
                for i_seq_p = 1:dk
                    s = s + q[(i - 1) * d + head_offset + i_seq_p] * k[(j - 1) * d + head_offset + i_seq_p]
                end
                scores[score_off + (i - 1) * n + j] = s * inv_sqrt_dk
            end

            # row-wise softmax of scores into probs
            # each row's softmax is independent of the others
            for i = 1:n
                row_max = scores[score_off + (i - 1) * n + 1]
                # running max requires a sequential scan; max() avoids a branch
                for i_seq_j = 2:n
                    row_max = max(row_max, scores[score_off + (i - 1) * n + i_seq_j])
                end
                # shifted exponentials are independent across columns
                for j = 1:n
                    kk = score_off + (i - 1) * n + j
                    probs[kk] = exp(scores[kk] - row_max)
                end
                row_sum = 0.0
                # normalizing sum requires a running total, so it stays sequential
                for i_seq_j = 1:n
                    row_sum = row_sum + probs[score_off + (i - 1) * n + i_seq_j]
                end
                # final division is independent across columns
                for j = 1:n
                    kk = score_off + (i - 1) * n + j
                    probs[kk] = probs[kk] / row_sum
                end
            end

            # ctx head slice = probs * v_head
            # every (token, head-dim) output entry is independent, and dk is fixed
            for idx3 = 1:n * dk
                i = div(idx3 - 1, dk) + 1
                p = mod(idx3 - 1, dk) + 1
                s = 0.0
                # summing over the attended tokens is a genuine accumulation
                for i_seq_j = 1:n
                    s = s + probs[score_off + (i - 1) * n + i_seq_j] * v[(i_seq_j - 1) * d + head_offset + p]
                end
                ctx[(i - 1) * d + head_offset + p] = s
            end
        end

        # --- output projection: attn_out = ctx * Wo + bo ---
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s = 0.0
            for i_seq_p = 1:d
                s = s + ctx[(i - 1) * d + i_seq_p] * wo[w_offset + (i_seq_p - 1) * d + j]
            end
            attn_out[(i - 1) * d + j] = s + bo[b_offset + j]
        end

        # --- first residual add: resid1 = x + attn_out ---
        for idx = 1:n_d
            resid1[idx] = x[idx] + attn_out[idx]
        end

        # --- first layer norm: normed1 = LayerNorm(resid1) ---
        # each row is normalized independently of the others
        for i = 1:n
            s = 0.0
            # mean requires a running sum, so this inner loop is sequential
            for i_seq_j = 1:d
                s = s + resid1[(i - 1) * d + i_seq_j]
            end
            row_mean = s / d
            s2 = 0.0
            # variance likewise accumulates sequentially
            for i_seq_j = 1:d
                diff = resid1[(i - 1) * d + i_seq_j] - row_mean
                s2 = s2 + diff * diff
            end
            row_var = s2 / d
            denom = sqrt(row_var + eps)
            # the final rescale is independent across columns
            for j = 1:d
                kk = (i - 1) * d + j
                normed1[kk] = (resid1[kk] - row_mean) / denom * ln1_gain[ln_offset + j] + ln1_bias[ln_offset + j]
            end
        end

        # --- feed-forward hidden layer: ff_hidden = relu(normed1 * W1 + b1) ---
        for idx = 1:n_dff
            i = div(idx - 1, dff) + 1
            j = mod(idx - 1, dff) + 1
            s = 0.0
            for i_seq_p = 1:d
                s = s + normed1[(i - 1) * d + i_seq_p] * w1[w1_offset + (i_seq_p - 1) * dff + j]
            end
            ff_hidden[(i - 1) * dff + j] = max(s + b1[b1_offset + j], 0.0)
        end

        # --- feed-forward output: ff_out = ff_hidden * W2 + b2 ---
        for idx = 1:n_d
            i = div(idx - 1, d) + 1
            j = mod(idx - 1, d) + 1
            s = 0.0
            for i_seq_p = 1:dff
                s = s + ff_hidden[(i - 1) * dff + i_seq_p] * w2[w2_offset + (i_seq_p - 1) * d + j]
            end
            ff_out[(i - 1) * d + j] = s + b2[b2_offset + j]
        end

        # --- second residual add: resid2 = normed1 + ff_out ---
        for idx = 1:n_d
            resid2[idx] = normed1[idx] + ff_out[idx]
        end

        # --- second layer norm: x_next = LayerNorm(resid2) ---
        for i = 1:n
            s = 0.0
            for i_seq_j = 1:d
                s = s + resid2[(i - 1) * d + i_seq_j]
            end
            row_mean = s / d
            s2 = 0.0
            for i_seq_j = 1:d
                diff = resid2[(i - 1) * d + i_seq_j] - row_mean
                s2 = s2 + diff * diff
            end
            row_var = s2 / d
            denom = sqrt(row_var + eps)
            for j = 1:d
                kk = (i - 1) * d + j
                x_next[kk] = (resid2[kk] - row_mean) / denom * ln2_gain[ln2_offset + j] + ln2_bias[ln2_offset + j]
            end
        end

        # --- carry this layer's output into x as the next layer's input ---
        for idx = 1:n_d
            x[idx] = x_next[idx]
        end
    end

    return nothing
end