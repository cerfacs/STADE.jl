import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_mpnn_hv_1!(agg, aggd, n_agg)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_agg - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    aggd[k] = 0.0
    agg[k] = 0.0
    return nothing
end

function cuda_kernel_mpnn_hv_2!(b_msg, b_msgd, dst, edge_feat, edge_featd, messages, messagesd, msg_input, msg_input_stack, msg_input_stack_d, msg_inputd, msg_scratch, msg_scratch_stack, msg_scratch_stack_d, msg_scratchd, n_edge_feat, n_edges, n_in_msg, n_msg_feat, n_node_feat, node_feat, node_featd, src, w_msg, w_msgd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_edges - 1, 1) + 1
        return nothing
    end
    e = 1 + (__tid - 1)
    s_node = src[e]
    d_node = dst[e]
    src_off = (s_node - 1) * n_node_feat
    dst_off = (d_node - 1) * n_node_feat
    edge_off = (e - 1) * n_edge_feat
    in_off = (e - 1) * n_in_msg
    msg_off = (e - 1) * n_msg_feat
    for k = 1:n_node_feat
        msg_input_stack_d[((e - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1] = msg_inputd[in_off + k]
        msg_input_stack[((e - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1] = msg_input[in_off + k]
        msg_inputd[in_off + k] = node_featd[src_off + k]
        msg_input[in_off + k] = node_feat[src_off + k]
    end
    for k = 1:n_node_feat
        msg_input_stack_d[(div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (((e - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1)] = msg_inputd[in_off + n_node_feat + k]
        msg_input_stack[(div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (((e - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1)] = msg_input[in_off + n_node_feat + k]
        msg_inputd[in_off + n_node_feat + k] = node_featd[dst_off + k]
        msg_input[in_off + n_node_feat + k] = node_feat[dst_off + k]
    end
    for k = 1:n_edge_feat
        msg_input_stack_d[((div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1)) + (((e - 1) * (div(n_edge_feat - 1, 1) + 1) + (k - 1)) + 1)] = msg_inputd[in_off + 2n_node_feat + k]
        msg_input_stack[((div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1)) + (((e - 1) * (div(n_edge_feat - 1, 1) + 1) + (k - 1)) + 1)] = msg_input[in_off + 2n_node_feat + k]
        msg_inputd[in_off + 2n_node_feat + k] = edge_featd[edge_off + k]
        msg_input[in_off + 2n_node_feat + k] = edge_feat[edge_off + k]
    end
    for o = 1:n_msg_feat
        sd = b_msgd[o]
        s = b_msg[o]
        for i_seq_i = 1:n_in_msg
            widx = (o - 1) * n_in_msg + i_seq_i
            sd = sd + (msg_input[in_off + i_seq_i] * w_msgd[widx] + w_msg[widx] * msg_inputd[in_off + i_seq_i])
            s = s + w_msg[widx] * msg_input[in_off + i_seq_i]
        end
        msg_scratch_stack_d[((e - 1) * (div(n_msg_feat - 1, 1) + 1) + (o - 1)) + 1] = msg_scratchd[msg_off + o]
        msg_scratch_stack[((e - 1) * (div(n_msg_feat - 1, 1) + 1) + (o - 1)) + 1] = msg_scratch[msg_off + o]
        msg_scratchd[msg_off + o] = sd
        msg_scratch[msg_off + o] = s
    end
    for k = 1:n_msg_feat
        msg_scratch_stack_d[(div(n_edges - 1, 1) + 1) * (div(n_msg_feat - 1, 1) + 1) + (((e - 1) * (div(n_msg_feat - 1, 1) + 1) + (k - 1)) + 1)] = msg_scratchd[msg_off + k]
        msg_scratch_stack[(div(n_edges - 1, 1) + 1) * (div(n_msg_feat - 1, 1) + 1) + (((e - 1) * (div(n_msg_feat - 1, 1) + 1) + (k - 1)) + 1)] = msg_scratch[msg_off + k]
        msg_scratchd[msg_off + k] = (0.5 * (1.0 + sign(msg_scratch[msg_off + k]))) * msg_scratchd[msg_off + k]
        msg_scratch[msg_off + k] = max(msg_scratch[msg_off + k], 0.0)
    end
    for k = 1:n_msg_feat
        messagesd[msg_off + k] = msg_scratchd[msg_off + k]
        messages[msg_off + k] = msg_scratch[msg_off + k]
    end
    return nothing
end

function cuda_kernel_mpnn_hv_3!(agg, aggd, dst, messages, messagesd, n_edges, n_msg_feat)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_edges - 1, 1) + 1
        return nothing
    end
    i_seq_e = 1 + (__tid - 1)
    d_node = dst[i_seq_e]
    msg_off = (i_seq_e - 1) * n_msg_feat
    agg_off = (d_node - 1) * n_msg_feat
    for j = 1:n_msg_feat
        CUDA.@atomic aggd[agg_off + j] += messagesd[msg_off + j]
        CUDA.@atomic agg[agg_off + j] += messages[msg_off + j]
    end
    return nothing
end

function cuda_kernel_mpnn_hv_4!(agg, aggd, b_upd, b_updd, n_in_upd, n_msg_feat, n_node_feat, n_nodes, node_feat, node_feat_out, node_feat_outd, node_featd, upd_input, upd_input_stack, upd_input_stack_d, upd_inputd, upd_scratch, upd_scratch_stack, upd_scratch_stack_d, upd_scratchd, w_upd, w_updd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_nodes - 1, 1) + 1
        return nothing
    end
    v = 1 + (__tid - 1)
    node_off = (v - 1) * n_node_feat
    agg_off = (v - 1) * n_msg_feat
    uin_off = (v - 1) * n_in_upd
    for k = 1:n_node_feat
        upd_input_stack_d[((v - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1] = upd_inputd[uin_off + k]
        upd_input_stack[((v - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1] = upd_input[uin_off + k]
        upd_inputd[uin_off + k] = node_featd[node_off + k]
        upd_input[uin_off + k] = node_feat[node_off + k]
    end
    for k = 1:n_msg_feat
        upd_input_stack_d[(div(n_nodes - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (((v - 1) * (div(n_msg_feat - 1, 1) + 1) + (k - 1)) + 1)] = upd_inputd[uin_off + n_node_feat + k]
        upd_input_stack[(div(n_nodes - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (((v - 1) * (div(n_msg_feat - 1, 1) + 1) + (k - 1)) + 1)] = upd_input[uin_off + n_node_feat + k]
        upd_inputd[uin_off + n_node_feat + k] = aggd[agg_off + k]
        upd_input[uin_off + n_node_feat + k] = agg[agg_off + k]
    end
    for o = 1:n_node_feat
        sd = b_updd[o]
        s = b_upd[o]
        for i_seq_i = 1:n_in_upd
            widx = (o - 1) * n_in_upd + i_seq_i
            sd = sd + (upd_input[uin_off + i_seq_i] * w_updd[widx] + w_upd[widx] * upd_inputd[uin_off + i_seq_i])
            s = s + w_upd[widx] * upd_input[uin_off + i_seq_i]
        end
        upd_scratch_stack_d[((v - 1) * (div(n_node_feat - 1, 1) + 1) + (o - 1)) + 1] = upd_scratchd[node_off + o]
        upd_scratch_stack[((v - 1) * (div(n_node_feat - 1, 1) + 1) + (o - 1)) + 1] = upd_scratch[node_off + o]
        upd_scratchd[node_off + o] = sd
        upd_scratch[node_off + o] = s
    end
    for k = 1:n_node_feat
        upd_scratch_stack_d[(div(n_nodes - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (((v - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1)] = upd_scratchd[node_off + k]
        upd_scratch_stack[(div(n_nodes - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (((v - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1)] = upd_scratch[node_off + k]
        upd_scratchd[node_off + k] = (0.5 * (1.0 + sign(upd_scratch[node_off + k]))) * upd_scratchd[node_off + k]
        upd_scratch[node_off + k] = max(upd_scratch[node_off + k], 0.0)
    end
    for k = 1:n_node_feat
        node_feat_outd[node_off + k] = upd_scratchd[node_off + k]
        node_feat_out[node_off + k] = upd_scratch[node_off + k]
    end
    return nothing
end

function cuda_kernel_mpnn_hv_5!(aggb, aggbd, b_updb, b_updbd, n_in_upd, n_msg_feat, n_node_feat, n_nodes, node_feat_outb, node_feat_outbd, node_featb, node_featbd, upd_input, upd_input_stack, upd_input_stack_d, upd_inputb, upd_inputbd, upd_inputd, upd_scratch, upd_scratch_stack, upd_scratch_stack_d, upd_scratchb, upd_scratchbd, upd_scratchd, w_upd, w_updb, w_updbd, w_updd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_nodes, -1) + 1
        return nothing
    end
    v = n_nodes + (__tid - 1) * -1
    node_off = (v - 1) * n_node_feat
    agg_off = (v - 1) * n_msg_feat
    uin_off = (v - 1) * n_in_upd
    for k = 1:n_node_feat
        upd_scratchbd[node_off + k] = upd_scratchbd[node_off + k] + node_feat_outbd[node_off + k]
        upd_scratchb[node_off + k] = upd_scratchb[node_off + k] + node_feat_outb[node_off + k]
        node_feat_outbd[node_off + k] = 0.0
        node_feat_outb[node_off + k] = 0.0
    end
    for k = n_node_feat:-1:1
        upd_scratchd[node_off + k] = upd_scratch_stack_d[(div(n_nodes - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (((v - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1)]
        upd_scratch[node_off + k] = upd_scratch_stack[(div(n_nodes - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (((v - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1)]
        upd_scratchbd[node_off + k] = (0.5 * (1.0 + sign(upd_scratch[node_off + k]))) * upd_scratchbd[node_off + k]
        upd_scratchb[node_off + k] = (0.5 * (1.0 + sign(upd_scratch[node_off + k]))) * upd_scratchb[node_off + k]
    end
    for o = n_node_feat:-1:1
        upd_scratchd[node_off + o] = upd_scratch_stack_d[((v - 1) * (div(n_node_feat - 1, 1) + 1) + (o - 1)) + 1]
        upd_scratch[node_off + o] = upd_scratch_stack[((v - 1) * (div(n_node_feat - 1, 1) + 1) + (o - 1)) + 1]
        sbd = sbd + upd_scratchbd[node_off + o]
        sb = sb + upd_scratchb[node_off + o]
        upd_scratchbd[node_off + o] = 0.0
        upd_scratchb[node_off + o] = 0.0
        for i_seq_i = n_in_upd:-1:1
            widx = (o - 1) * n_in_upd + i_seq_i
            CUDA.@atomic w_updbd[widx] += sb * upd_inputd[uin_off + i_seq_i] + upd_input[uin_off + i_seq_i] * sbd
            CUDA.@atomic w_updb[widx] += upd_input[uin_off + i_seq_i] * sb
            upd_inputbd[uin_off + i_seq_i] = upd_inputbd[uin_off + i_seq_i] + (sb * w_updd[widx] + w_upd[widx] * sbd)
            upd_inputb[uin_off + i_seq_i] = upd_inputb[uin_off + i_seq_i] + w_upd[widx] * sb
        end
        CUDA.@atomic b_updbd[o] += sbd
        CUDA.@atomic b_updb[o] += sb
        sbd = 0.0
        sb = 0.0
    end
    for k = n_msg_feat:-1:1
        upd_inputd[uin_off + n_node_feat + k] = upd_input_stack_d[(div(n_nodes - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (((v - 1) * (div(n_msg_feat - 1, 1) + 1) + (k - 1)) + 1)]
        upd_input[uin_off + n_node_feat + k] = upd_input_stack[(div(n_nodes - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (((v - 1) * (div(n_msg_feat - 1, 1) + 1) + (k - 1)) + 1)]
        aggbd[agg_off + k] = aggbd[agg_off + k] + upd_inputbd[uin_off + n_node_feat + k]
        aggb[agg_off + k] = aggb[agg_off + k] + upd_inputb[uin_off + n_node_feat + k]
        upd_inputbd[uin_off + n_node_feat + k] = 0.0
        upd_inputb[uin_off + n_node_feat + k] = 0.0
    end
    for k = n_node_feat:-1:1
        upd_inputd[uin_off + k] = upd_input_stack_d[((v - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1]
        upd_input[uin_off + k] = upd_input_stack[((v - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1]
        node_featbd[node_off + k] = node_featbd[node_off + k] + upd_inputbd[uin_off + k]
        node_featb[node_off + k] = node_featb[node_off + k] + upd_inputb[uin_off + k]
        upd_inputbd[uin_off + k] = 0.0
        upd_inputb[uin_off + k] = 0.0
    end
    return nothing
end

function cuda_kernel_mpnn_hv_6!(aggb, aggbd, dst, messagesb, messagesbd, n_edges, n_msg_feat)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_edges, -1) + 1
        return nothing
    end
    i_seq_e = n_edges + (__tid - 1) * -1
    d_node = dst[i_seq_e]
    msg_off = (i_seq_e - 1) * n_msg_feat
    agg_off = (d_node - 1) * n_msg_feat
    for j = 1:n_msg_feat
        messagesbd[msg_off + j] = messagesbd[msg_off + j] + aggbd[agg_off + j]
        messagesb[msg_off + j] = messagesb[msg_off + j] + aggb[agg_off + j]
    end
    return nothing
end

function cuda_kernel_mpnn_hv_7!(b_msgb, b_msgbd, dst, edge_featb, edge_featbd, messagesb, messagesbd, msg_input, msg_input_stack, msg_input_stack_d, msg_inputb, msg_inputbd, msg_inputd, msg_scratch, msg_scratch_stack, msg_scratch_stack_d, msg_scratchb, msg_scratchbd, msg_scratchd, n_edge_feat, n_edges, n_in_msg, n_msg_feat, n_node_feat, node_featb, node_featbd, src, w_msg, w_msgb, w_msgbd, w_msgd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - n_edges, -1) + 1
        return nothing
    end
    e = n_edges + (__tid - 1) * -1
    s_node = src[e]
    d_node = dst[e]
    src_off = (s_node - 1) * n_node_feat
    dst_off = (d_node - 1) * n_node_feat
    edge_off = (e - 1) * n_edge_feat
    in_off = (e - 1) * n_in_msg
    msg_off = (e - 1) * n_msg_feat
    for k = 1:n_msg_feat
        msg_scratchbd[msg_off + k] = msg_scratchbd[msg_off + k] + messagesbd[msg_off + k]
        msg_scratchb[msg_off + k] = msg_scratchb[msg_off + k] + messagesb[msg_off + k]
        messagesbd[msg_off + k] = 0.0
        messagesb[msg_off + k] = 0.0
    end
    for k = n_msg_feat:-1:1
        msg_scratchd[msg_off + k] = msg_scratch_stack_d[(div(n_edges - 1, 1) + 1) * (div(n_msg_feat - 1, 1) + 1) + (((e - 1) * (div(n_msg_feat - 1, 1) + 1) + (k - 1)) + 1)]
        msg_scratch[msg_off + k] = msg_scratch_stack[(div(n_edges - 1, 1) + 1) * (div(n_msg_feat - 1, 1) + 1) + (((e - 1) * (div(n_msg_feat - 1, 1) + 1) + (k - 1)) + 1)]
        msg_scratchbd[msg_off + k] = (0.5 * (1.0 + sign(msg_scratch[msg_off + k]))) * msg_scratchbd[msg_off + k]
        msg_scratchb[msg_off + k] = (0.5 * (1.0 + sign(msg_scratch[msg_off + k]))) * msg_scratchb[msg_off + k]
    end
    for o = n_msg_feat:-1:1
        msg_scratchd[msg_off + o] = msg_scratch_stack_d[((e - 1) * (div(n_msg_feat - 1, 1) + 1) + (o - 1)) + 1]
        msg_scratch[msg_off + o] = msg_scratch_stack[((e - 1) * (div(n_msg_feat - 1, 1) + 1) + (o - 1)) + 1]
        sbd = sbd + msg_scratchbd[msg_off + o]
        sb = sb + msg_scratchb[msg_off + o]
        msg_scratchbd[msg_off + o] = 0.0
        msg_scratchb[msg_off + o] = 0.0
        for i_seq_i = n_in_msg:-1:1
            widx = (o - 1) * n_in_msg + i_seq_i
            CUDA.@atomic w_msgbd[widx] += sb * msg_inputd[in_off + i_seq_i] + msg_input[in_off + i_seq_i] * sbd
            CUDA.@atomic w_msgb[widx] += msg_input[in_off + i_seq_i] * sb
            msg_inputbd[in_off + i_seq_i] = msg_inputbd[in_off + i_seq_i] + (sb * w_msgd[widx] + w_msg[widx] * sbd)
            msg_inputb[in_off + i_seq_i] = msg_inputb[in_off + i_seq_i] + w_msg[widx] * sb
        end
        CUDA.@atomic b_msgbd[o] += sbd
        CUDA.@atomic b_msgb[o] += sb
        sbd = 0.0
        sb = 0.0
    end
    for k = n_edge_feat:-1:1
        msg_inputd[in_off + 2n_node_feat + k] = msg_input_stack_d[((div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1)) + (((e - 1) * (div(n_edge_feat - 1, 1) + 1) + (k - 1)) + 1)]
        msg_input[in_off + 2n_node_feat + k] = msg_input_stack[((div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1)) + (((e - 1) * (div(n_edge_feat - 1, 1) + 1) + (k - 1)) + 1)]
        edge_featbd[edge_off + k] = edge_featbd[edge_off + k] + msg_inputbd[in_off + 2n_node_feat + k]
        edge_featb[edge_off + k] = edge_featb[edge_off + k] + msg_inputb[in_off + 2n_node_feat + k]
        msg_inputbd[in_off + 2n_node_feat + k] = 0.0
        msg_inputb[in_off + 2n_node_feat + k] = 0.0
    end
    for k = n_node_feat:-1:1
        msg_inputd[in_off + n_node_feat + k] = msg_input_stack_d[(div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (((e - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1)]
        msg_input[in_off + n_node_feat + k] = msg_input_stack[(div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (((e - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1)]
        CUDA.@atomic node_featbd[dst_off + k] += msg_inputbd[in_off + n_node_feat + k]
        CUDA.@atomic node_featb[dst_off + k] += msg_inputb[in_off + n_node_feat + k]
        msg_inputbd[in_off + n_node_feat + k] = 0.0
        msg_inputb[in_off + n_node_feat + k] = 0.0
    end
    for k = n_node_feat:-1:1
        msg_inputd[in_off + k] = msg_input_stack_d[((e - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1]
        msg_input[in_off + k] = msg_input_stack[((e - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1]
        CUDA.@atomic node_featbd[src_off + k] += msg_inputbd[in_off + k]
        CUDA.@atomic node_featb[src_off + k] += msg_inputb[in_off + k]
        msg_inputbd[in_off + k] = 0.0
        msg_inputb[in_off + k] = 0.0
    end
    return nothing
end

function cuda_kernel_mpnn_hv_8!(aggb, aggbd, n_agg)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_agg - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    aggbd[k] = 0.0
    aggb[k] = 0.0
    return nothing
end

function cuda_kernel_mpnn_1!(agg, n_agg)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_agg - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    agg[k] = 0.0
    return nothing
end

function cuda_kernel_mpnn_2!(b_msg, dst, edge_feat, messages, msg_input, msg_scratch, n_edge_feat, n_edges, n_in_msg, n_msg_feat, n_node_feat, node_feat, src, w_msg)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_edges - 1, 1) + 1
        return nothing
    end
    e = 1 + (__tid - 1)
    s_node = src[e]
    d_node = dst[e]
    src_off = (s_node - 1) * n_node_feat
    dst_off = (d_node - 1) * n_node_feat
    edge_off = (e - 1) * n_edge_feat
    in_off = (e - 1) * n_in_msg
    msg_off = (e - 1) * n_msg_feat
    for k = 1:n_node_feat
        msg_input[in_off + k] = node_feat[src_off + k]
    end
    for k = 1:n_node_feat
        msg_input[in_off + n_node_feat + k] = node_feat[dst_off + k]
    end
    for k = 1:n_edge_feat
        msg_input[in_off + 2n_node_feat + k] = edge_feat[edge_off + k]
    end
    for o = 1:n_msg_feat
        s = b_msg[o]
        for i_seq_i = 1:n_in_msg
            widx = (o - 1) * n_in_msg + i_seq_i
            s = s + w_msg[widx] * msg_input[in_off + i_seq_i]
        end
        msg_scratch[msg_off + o] = s
    end
    for k = 1:n_msg_feat
        msg_scratch[msg_off + k] = max(msg_scratch[msg_off + k], 0.0)
    end
    for k = 1:n_msg_feat
        messages[msg_off + k] = msg_scratch[msg_off + k]
    end
    return nothing
end

function cuda_kernel_mpnn_3!(agg, dst, messages, n_edges, n_msg_feat)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_edges - 1, 1) + 1
        return nothing
    end
    i_seq_e = 1 + (__tid - 1)
    d_node = dst[i_seq_e]
    msg_off = (i_seq_e - 1) * n_msg_feat
    agg_off = (d_node - 1) * n_msg_feat
    for j = 1:n_msg_feat
        CUDA.@atomic agg[agg_off + j] += messages[msg_off + j]
    end
    return nothing
end

function cuda_kernel_mpnn_4!(agg, b_upd, n_in_upd, n_msg_feat, n_node_feat, n_nodes, node_feat, node_feat_out, upd_input, upd_scratch, w_upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(n_nodes - 1, 1) + 1
        return nothing
    end
    v = 1 + (__tid - 1)
    node_off = (v - 1) * n_node_feat
    agg_off = (v - 1) * n_msg_feat
    uin_off = (v - 1) * n_in_upd
    for k = 1:n_node_feat
        upd_input[uin_off + k] = node_feat[node_off + k]
    end
    for k = 1:n_msg_feat
        upd_input[uin_off + n_node_feat + k] = agg[agg_off + k]
    end
    for o = 1:n_node_feat
        s = b_upd[o]
        for i_seq_i = 1:n_in_upd
            widx = (o - 1) * n_in_upd + i_seq_i
            s = s + w_upd[widx] * upd_input[uin_off + i_seq_i]
        end
        upd_scratch[node_off + o] = s
    end
    for k = 1:n_node_feat
        upd_scratch[node_off + k] = max(upd_scratch[node_off + k], 0.0)
    end
    for k = 1:n_node_feat
        node_feat_out[node_off + k] = upd_scratch[node_off + k]
    end
    return nothing
end

function initstacks_mpnn_b_cuda(n_edge_feat, n_edges, n_msg_feat, n_node_feat, n_nodes)
    msg_input_stack = CuArray{Float64}(undef, ((div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1)) + (div(n_edges - 1, 1) + 1) * (div(n_edge_feat - 1, 1) + 1))
    msg_scratch_stack = CuArray{Float64}(undef, (div(n_edges - 1, 1) + 1) * (div(n_msg_feat - 1, 1) + 1) + (div(n_edges - 1, 1) + 1) * (div(n_msg_feat - 1, 1) + 1))
    upd_input_stack = CuArray{Float64}(undef, (div(n_nodes - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (div(n_nodes - 1, 1) + 1) * (div(n_msg_feat - 1, 1) + 1))
    upd_scratch_stack = CuArray{Float64}(undef, (div(n_nodes - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (div(n_nodes - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1))
    return (msg_input_stack, msg_scratch_stack, upd_input_stack, upd_scratch_stack)
end

function mpnn_hv_cuda(node_feat, node_featb, edge_feat, edge_featb, src, dst, w_msg, w_msgb, b_msg, b_msgb, w_upd, w_updb, b_upd, b_updb, n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat, msg_input, msg_inputb, msg_scratch, msg_scratchb, messages, messagesb, agg, aggb, upd_input, upd_inputb, upd_scratch, upd_scratchb, node_feat_out, node_feat_outb, node_featd, node_featbd, edge_featd, edge_featbd, w_msgd, w_msgbd, b_msgd, b_msgbd, w_updd, w_updbd, b_updd, b_updbd, msg_inputd, msg_inputbd, msg_scratchd, msg_scratchbd, messagesd, messagesbd, aggd, aggbd, upd_inputd, upd_inputbd, upd_scratchd, upd_scratchbd, node_feat_outd, node_feat_outbd, msg_input_stack, msg_scratch_stack, upd_input_stack, upd_scratch_stack)
    nthread_per_block = 256
    msg_input_stack_d = CuArray{Float64}(undef, length(msg_input_stack))
    msg_scratch_stack_d = CuArray{Float64}(undef, length(msg_scratch_stack))
    upd_input_stack_d = CuArray{Float64}(undef, length(upd_input_stack))
    upd_scratch_stack_d = CuArray{Float64}(undef, length(upd_scratch_stack))
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    n_in_msg = 2n_node_feat + n_edge_feat
    n_in_upd = n_node_feat + n_msg_feat
    n_agg = n_nodes * n_msg_feat
    @cuda threads = nthread_per_block blocks = cld(div(n_agg - 1, 1) + 1, nthread_per_block) cuda_kernel_mpnn_hv_1!(agg, aggd, n_agg)
    @cuda threads = nthread_per_block blocks = cld(div(n_edges - 1, 1) + 1, nthread_per_block) cuda_kernel_mpnn_hv_2!(b_msg, b_msgd, dst, edge_feat, edge_featd, messages, messagesd, msg_input, msg_input_stack, msg_input_stack_d, msg_inputd, msg_scratch, msg_scratch_stack, msg_scratch_stack_d, msg_scratchd, n_edge_feat, n_edges, n_in_msg, n_msg_feat, n_node_feat, node_feat, node_featd, src, w_msg, w_msgd)
    @cuda threads = nthread_per_block blocks = cld(div(n_edges - 1, 1) + 1, nthread_per_block) cuda_kernel_mpnn_hv_3!(agg, aggd, dst, messages, messagesd, n_edges, n_msg_feat)
    @cuda threads = nthread_per_block blocks = cld(div(n_nodes - 1, 1) + 1, nthread_per_block) cuda_kernel_mpnn_hv_4!(agg, aggd, b_upd, b_updd, n_in_upd, n_msg_feat, n_node_feat, n_nodes, node_feat, node_feat_out, node_feat_outd, node_featd, upd_input, upd_input_stack, upd_input_stack_d, upd_inputd, upd_scratch, upd_scratch_stack, upd_scratch_stack_d, upd_scratchd, w_upd, w_updd)
    n_in_msg = 2n_node_feat + n_edge_feat
    n_in_upd = n_node_feat + n_msg_feat
    n_agg = n_nodes * n_msg_feat
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_nodes, -1) + 1, nthread_per_block) cuda_kernel_mpnn_hv_5!(aggb, aggbd, b_updb, b_updbd, n_in_upd, n_msg_feat, n_node_feat, n_nodes, node_feat_outb, node_feat_outbd, node_featb, node_featbd, upd_input, upd_input_stack, upd_input_stack_d, upd_inputb, upd_inputbd, upd_inputd, upd_scratch, upd_scratch_stack, upd_scratch_stack_d, upd_scratchb, upd_scratchbd, upd_scratchd, w_upd, w_updb, w_updbd, w_updd)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_edges, -1) + 1, nthread_per_block) cuda_kernel_mpnn_hv_6!(aggb, aggbd, dst, messagesb, messagesbd, n_edges, n_msg_feat)
    @cuda threads = nthread_per_block blocks = cld(div(1 - n_edges, -1) + 1, nthread_per_block) cuda_kernel_mpnn_hv_7!(b_msgb, b_msgbd, dst, edge_featb, edge_featbd, messagesb, messagesbd, msg_input, msg_input_stack, msg_input_stack_d, msg_inputb, msg_inputbd, msg_inputd, msg_scratch, msg_scratch_stack, msg_scratch_stack_d, msg_scratchb, msg_scratchbd, msg_scratchd, n_edge_feat, n_edges, n_in_msg, n_msg_feat, n_node_feat, node_featb, node_featbd, src, w_msg, w_msgb, w_msgbd, w_msgd)
    @cuda threads = nthread_per_block blocks = cld(div(n_agg - 1, 1) + 1, nthread_per_block) cuda_kernel_mpnn_hv_8!(aggb, aggbd, n_agg)
    return nothing
end

function mpnn_cuda(node_feat, edge_feat, src, dst, w_msg, b_msg, w_upd, b_upd, n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat, msg_input, msg_scratch, messages, agg, upd_input, upd_scratch, node_feat_out)
    nthread_per_block = 256
    n_in_msg = 2n_node_feat + n_edge_feat
    n_in_upd = n_node_feat + n_msg_feat
    n_agg = n_nodes * n_msg_feat
    @cuda threads = nthread_per_block blocks = cld(div(n_agg - 1, 1) + 1, nthread_per_block) cuda_kernel_mpnn_1!(agg, n_agg)
    @cuda threads = nthread_per_block blocks = cld(div(n_edges - 1, 1) + 1, nthread_per_block) cuda_kernel_mpnn_2!(b_msg, dst, edge_feat, messages, msg_input, msg_scratch, n_edge_feat, n_edges, n_in_msg, n_msg_feat, n_node_feat, node_feat, src, w_msg)
    @cuda threads = nthread_per_block blocks = cld(div(n_edges - 1, 1) + 1, nthread_per_block) cuda_kernel_mpnn_3!(agg, dst, messages, n_edges, n_msg_feat)
    @cuda threads = nthread_per_block blocks = cld(div(n_nodes - 1, 1) + 1, nthread_per_block) cuda_kernel_mpnn_4!(agg, b_upd, n_in_upd, n_msg_feat, n_node_feat, n_nodes, node_feat, node_feat_out, upd_input, upd_scratch, w_upd)
    return nothing
end
