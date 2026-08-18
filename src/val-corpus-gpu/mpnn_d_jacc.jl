import Pkg
haskey(Pkg.project().dependencies, "JACC") || Pkg.add(name = "JACC", version = "1")
haskey(Pkg.project().dependencies, "Atomix") || Pkg.add("Atomix")
import JACC
import Atomix
JACC.@init_backend

function jacc_kernel_mpnn_d_1!(__jacc_i, agg, aggd, n_agg)
    k = 1 + (__jacc_i - 1)
    aggd[k] = 0.0
    agg[k] = 0.0
    return nothing
end

function jacc_kernel_mpnn_d_2!(__jacc_i, b_msg, b_msgd, dst, edge_feat, edge_featd, messages, messagesd, msg_input, msg_inputd, msg_scratch, msg_scratchd, n_edge_feat, n_edges, n_in_msg, n_msg_feat, n_node_feat, node_feat, node_featd, src, w_msg, w_msgd)
    e = 1 + (__jacc_i - 1)
    s_noded = 0.0
    s_node = src[e]
    d_noded = 0.0
    d_node = dst[e]
    src_offd = 0.0
    src_off = (s_node - 1) * n_node_feat
    dst_offd = 0.0
    dst_off = (d_node - 1) * n_node_feat
    edge_offd = 0.0
    edge_off = (e - 1) * n_edge_feat
    in_offd = 0.0
    in_off = (e - 1) * n_in_msg
    msg_offd = 0.0
    msg_off = (e - 1) * n_msg_feat
    for k = 1:n_node_feat
        msg_inputd[in_off + k] = node_featd[src_off + k]
        msg_input[in_off + k] = node_feat[src_off + k]
    end
    for k = 1:n_node_feat
        msg_inputd[in_off + n_node_feat + k] = node_featd[dst_off + k]
        msg_input[in_off + n_node_feat + k] = node_feat[dst_off + k]
    end
    for k = 1:n_edge_feat
        msg_inputd[in_off + 2n_node_feat + k] = edge_featd[edge_off + k]
        msg_input[in_off + 2n_node_feat + k] = edge_feat[edge_off + k]
    end
    for o = 1:n_msg_feat
        sd = b_msgd[o]
        s = b_msg[o]
        for i_seq_i = 1:n_in_msg
            widxd = 0.0
            widx = (o - 1) * n_in_msg + i_seq_i
            sd = sd + (msg_input[in_off + i_seq_i] * w_msgd[widx] + w_msg[widx] * msg_inputd[in_off + i_seq_i])
            s = s + w_msg[widx] * msg_input[in_off + i_seq_i]
        end
        msg_scratchd[msg_off + o] = sd
        msg_scratch[msg_off + o] = s
    end
    for k = 1:n_msg_feat
        msg_scratchd[msg_off + k] = (0.5 * (1.0 + sign(msg_scratch[msg_off + k]))) * msg_scratchd[msg_off + k]
        msg_scratch[msg_off + k] = max(msg_scratch[msg_off + k], 0.0)
    end
    for k = 1:n_msg_feat
        messagesd[msg_off + k] = msg_scratchd[msg_off + k]
        messages[msg_off + k] = msg_scratch[msg_off + k]
    end
    return nothing
end

function jacc_kernel_mpnn_d_3!(__jacc_i, agg, aggd, dst, messages, messagesd, n_edges, n_msg_feat)
    i_seq_e = 1 + (__jacc_i - 1)
    d_noded = 0.0
    d_node = dst[i_seq_e]
    msg_offd = 0.0
    msg_off = (i_seq_e - 1) * n_msg_feat
    agg_offd = 0.0
    agg_off = (d_node - 1) * n_msg_feat
    for j = 1:n_msg_feat
        Atomix.@atomic aggd[agg_off + j] += messagesd[msg_off + j]
        Atomix.@atomic agg[agg_off + j] += messages[msg_off + j]
    end
    return nothing
end

function jacc_kernel_mpnn_d_4!(__jacc_i, agg, aggd, b_upd, b_updd, n_in_upd, n_msg_feat, n_node_feat, n_nodes, node_feat, node_feat_out, node_feat_outd, node_featd, upd_input, upd_inputd, upd_scratch, upd_scratchd, w_upd, w_updd)
    v = 1 + (__jacc_i - 1)
    node_offd = 0.0
    node_off = (v - 1) * n_node_feat
    agg_offd = 0.0
    agg_off = (v - 1) * n_msg_feat
    uin_offd = 0.0
    uin_off = (v - 1) * n_in_upd
    for k = 1:n_node_feat
        upd_inputd[uin_off + k] = node_featd[node_off + k]
        upd_input[uin_off + k] = node_feat[node_off + k]
    end
    for k = 1:n_msg_feat
        upd_inputd[uin_off + n_node_feat + k] = aggd[agg_off + k]
        upd_input[uin_off + n_node_feat + k] = agg[agg_off + k]
    end
    for o = 1:n_node_feat
        sd = b_updd[o]
        s = b_upd[o]
        for i_seq_i = 1:n_in_upd
            widxd = 0.0
            widx = (o - 1) * n_in_upd + i_seq_i
            sd = sd + (upd_input[uin_off + i_seq_i] * w_updd[widx] + w_upd[widx] * upd_inputd[uin_off + i_seq_i])
            s = s + w_upd[widx] * upd_input[uin_off + i_seq_i]
        end
        upd_scratchd[node_off + o] = sd
        upd_scratch[node_off + o] = s
    end
    for k = 1:n_node_feat
        upd_scratchd[node_off + k] = (0.5 * (1.0 + sign(upd_scratch[node_off + k]))) * upd_scratchd[node_off + k]
        upd_scratch[node_off + k] = max(upd_scratch[node_off + k], 0.0)
    end
    for k = 1:n_node_feat
        node_feat_outd[node_off + k] = upd_scratchd[node_off + k]
        node_feat_out[node_off + k] = upd_scratch[node_off + k]
    end
    return nothing
end

function jacc_kernel_mpnn_1!(__jacc_i, agg, n_agg)
    k = 1 + (__jacc_i - 1)
    agg[k] = 0.0
    return nothing
end

function jacc_kernel_mpnn_2!(__jacc_i, b_msg, dst, edge_feat, messages, msg_input, msg_scratch, n_edge_feat, n_edges, n_in_msg, n_msg_feat, n_node_feat, node_feat, src, w_msg)
    e = 1 + (__jacc_i - 1)
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

function jacc_kernel_mpnn_3!(__jacc_i, agg, dst, messages, n_edges, n_msg_feat)
    i_seq_e = 1 + (__jacc_i - 1)
    d_node = dst[i_seq_e]
    msg_off = (i_seq_e - 1) * n_msg_feat
    agg_off = (d_node - 1) * n_msg_feat
    for j = 1:n_msg_feat
        Atomix.@atomic agg[agg_off + j] += messages[msg_off + j]
    end
    return nothing
end

function jacc_kernel_mpnn_4!(__jacc_i, agg, b_upd, n_in_upd, n_msg_feat, n_node_feat, n_nodes, node_feat, node_feat_out, upd_input, upd_scratch, w_upd)
    v = 1 + (__jacc_i - 1)
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

function mpnn_d_jacc(node_feat, node_featd, edge_feat, edge_featd, src, dst, w_msg, w_msgd, b_msg, b_msgd, w_upd, w_updd, b_upd, b_updd, n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat, msg_input, msg_inputd, msg_scratch, msg_scratchd, messages, messagesd, agg, aggd, upd_input, upd_inputd, upd_scratch, upd_scratchd, node_feat_out, node_feat_outd)
    n_in_msgd = 0.0
    n_in_msg = 2n_node_feat + n_edge_feat
    n_in_updd = 0.0
    n_in_upd = n_node_feat + n_msg_feat
    n_aggd = 0.0
    n_agg = n_nodes * n_msg_feat
    JACC.@parallel_for range = div(n_agg - 1, 1) + 1 jacc_kernel_mpnn_d_1!(agg, aggd, n_agg)
    JACC.@parallel_for range = div(n_edges - 1, 1) + 1 jacc_kernel_mpnn_d_2!(b_msg, b_msgd, dst, edge_feat, edge_featd, messages, messagesd, msg_input, msg_inputd, msg_scratch, msg_scratchd, n_edge_feat, n_edges, n_in_msg, n_msg_feat, n_node_feat, node_feat, node_featd, src, w_msg, w_msgd)
    JACC.@parallel_for range = div(n_edges - 1, 1) + 1 jacc_kernel_mpnn_d_3!(agg, aggd, dst, messages, messagesd, n_edges, n_msg_feat)
    JACC.@parallel_for range = div(n_nodes - 1, 1) + 1 jacc_kernel_mpnn_d_4!(agg, aggd, b_upd, b_updd, n_in_upd, n_msg_feat, n_node_feat, n_nodes, node_feat, node_feat_out, node_feat_outd, node_featd, upd_input, upd_inputd, upd_scratch, upd_scratchd, w_upd, w_updd)
    return nothing
end

function mpnn_jacc(node_feat, edge_feat, src, dst, w_msg, b_msg, w_upd, b_upd, n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat, msg_input, msg_scratch, messages, agg, upd_input, upd_scratch, node_feat_out)
    n_in_msg = 2n_node_feat + n_edge_feat
    n_in_upd = n_node_feat + n_msg_feat
    n_agg = n_nodes * n_msg_feat
    JACC.@parallel_for range = div(n_agg - 1, 1) + 1 jacc_kernel_mpnn_1!(agg, n_agg)
    JACC.@parallel_for range = div(n_edges - 1, 1) + 1 jacc_kernel_mpnn_2!(b_msg, dst, edge_feat, messages, msg_input, msg_scratch, n_edge_feat, n_edges, n_in_msg, n_msg_feat, n_node_feat, node_feat, src, w_msg)
    JACC.@parallel_for range = div(n_edges - 1, 1) + 1 jacc_kernel_mpnn_3!(agg, dst, messages, n_edges, n_msg_feat)
    JACC.@parallel_for range = div(n_nodes - 1, 1) + 1 jacc_kernel_mpnn_4!(agg, b_upd, n_in_upd, n_msg_feat, n_node_feat, n_nodes, node_feat, node_feat_out, upd_input, upd_scratch, w_upd)
    return nothing
end
