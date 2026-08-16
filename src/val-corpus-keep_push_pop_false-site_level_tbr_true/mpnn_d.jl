function mpnn_d(node_feat, node_featd, edge_feat, edge_featd, src, dst, w_msg, w_msgd, b_msg, b_msgd, w_upd, w_updd, b_upd, b_updd, n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat, msg_input, msg_inputd, msg_scratch, msg_scratchd, messages, messagesd, agg, aggd, upd_input, upd_inputd, upd_scratch, upd_scratchd, node_feat_out, node_feat_outd)
    zero_offd = 0.0
    zero_off = 0
    n_in_msgd = 0.0
    n_in_msg = 2n_node_feat + n_edge_feat
    n_in_updd = 0.0
    n_in_upd = n_node_feat + n_msg_feat
    n_aggd = 0.0
    n_agg = n_nodes * n_msg_feat
    for k = 1:n_agg
        aggd[k] = 0.0
        agg[k] = 0.0
    end
    for e = 1:n_edges
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
        for k = 1:n_node_feat
            msg_inputd[zero_off + k] = node_featd[src_off + k]
            msg_input[zero_off + k] = node_feat[src_off + k]
        end
        for k = 1:n_node_feat
            msg_inputd[n_node_feat + k] = node_featd[dst_off + k]
            msg_input[n_node_feat + k] = node_feat[dst_off + k]
        end
        for k = 1:n_edge_feat
            msg_inputd[2n_node_feat + k] = edge_featd[edge_off + k]
            msg_input[2n_node_feat + k] = edge_feat[edge_off + k]
        end
        for o = 1:n_msg_feat
            sd = b_msgd[o]
            s = b_msg[o]
            for i_seq_i = 1:n_in_msg
                widxd = 0.0
                widx = (o - 1) * n_in_msg + i_seq_i
                sd = sd + (msg_input[i_seq_i] * w_msgd[widx] + w_msg[widx] * msg_inputd[i_seq_i])
                s = s + w_msg[widx] * msg_input[i_seq_i]
            end
            msg_scratchd[o] = sd
            msg_scratch[o] = s
        end
        for k = 1:n_msg_feat
            msg_scratchd[k] = (0.5 * (1.0 + sign(msg_scratch[k]))) * msg_scratchd[k]
            msg_scratch[k] = max(msg_scratch[k], 0.0)
        end
        msg_offd = 0.0
        msg_off = (e - 1) * n_msg_feat
        for k = 1:n_msg_feat
            messagesd[msg_off + k] = msg_scratchd[zero_off + k]
            messages[msg_off + k] = msg_scratch[zero_off + k]
        end
    end
    for i_seq_e = 1:n_edges
        d_noded = 0.0
        d_node = dst[i_seq_e]
        msg_offd = 0.0
        msg_off = (i_seq_e - 1) * n_msg_feat
        agg_offd = 0.0
        agg_off = (d_node - 1) * n_msg_feat
        for j = 1:n_msg_feat
            aggd[agg_off + j] = aggd[agg_off + j] + messagesd[msg_off + j]
            agg[agg_off + j] = agg[agg_off + j] + messages[msg_off + j]
        end
    end
    for v = 1:n_nodes
        node_offd = 0.0
        node_off = (v - 1) * n_node_feat
        agg_offd = 0.0
        agg_off = (v - 1) * n_msg_feat
        for k = 1:n_node_feat
            upd_inputd[zero_off + k] = node_featd[node_off + k]
            upd_input[zero_off + k] = node_feat[node_off + k]
        end
        for k = 1:n_msg_feat
            upd_inputd[n_node_feat + k] = aggd[agg_off + k]
            upd_input[n_node_feat + k] = agg[agg_off + k]
        end
        for o = 1:n_node_feat
            sd = b_updd[o]
            s = b_upd[o]
            for i_seq_i = 1:n_in_upd
                widxd = 0.0
                widx = (o - 1) * n_in_upd + i_seq_i
                sd = sd + (upd_input[i_seq_i] * w_updd[widx] + w_upd[widx] * upd_inputd[i_seq_i])
                s = s + w_upd[widx] * upd_input[i_seq_i]
            end
            upd_scratchd[o] = sd
            upd_scratch[o] = s
        end
        for k = 1:n_node_feat
            upd_scratchd[k] = (0.5 * (1.0 + sign(upd_scratch[k]))) * upd_scratchd[k]
            upd_scratch[k] = max(upd_scratch[k], 0.0)
        end
        for k = 1:n_node_feat
            node_feat_outd[node_off + k] = upd_scratchd[zero_off + k]
            node_feat_out[node_off + k] = upd_scratch[zero_off + k]
        end
    end
    return nothing
end

function mpnn(node_feat, edge_feat, src, dst, w_msg, b_msg, w_upd, b_upd, n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat, msg_input, msg_scratch, messages, agg, upd_input, upd_scratch, node_feat_out)
    zero_off = 0
    n_in_msg = 2n_node_feat + n_edge_feat
    n_in_upd = n_node_feat + n_msg_feat
    n_agg = n_nodes * n_msg_feat
    for k = 1:n_agg
        agg[k] = 0.0
    end
    for e = 1:n_edges
        s_node = src[e]
        d_node = dst[e]
        src_off = (s_node - 1) * n_node_feat
        dst_off = (d_node - 1) * n_node_feat
        edge_off = (e - 1) * n_edge_feat
        for k = 1:n_node_feat
            msg_input[zero_off + k] = node_feat[src_off + k]
        end
        for k = 1:n_node_feat
            msg_input[n_node_feat + k] = node_feat[dst_off + k]
        end
        for k = 1:n_edge_feat
            msg_input[2n_node_feat + k] = edge_feat[edge_off + k]
        end
        for o = 1:n_msg_feat
            s = b_msg[o]
            for i_seq_i = 1:n_in_msg
                widx = (o - 1) * n_in_msg + i_seq_i
                s = s + w_msg[widx] * msg_input[i_seq_i]
            end
            msg_scratch[o] = s
        end
        for k = 1:n_msg_feat
            msg_scratch[k] = max(msg_scratch[k], 0.0)
        end
        msg_off = (e - 1) * n_msg_feat
        for k = 1:n_msg_feat
            messages[msg_off + k] = msg_scratch[zero_off + k]
        end
    end
    for i_seq_e = 1:n_edges
        d_node = dst[i_seq_e]
        msg_off = (i_seq_e - 1) * n_msg_feat
        agg_off = (d_node - 1) * n_msg_feat
        for j = 1:n_msg_feat
            agg[agg_off + j] = agg[agg_off + j] + messages[msg_off + j]
        end
    end
    for v = 1:n_nodes
        node_off = (v - 1) * n_node_feat
        agg_off = (v - 1) * n_msg_feat
        for k = 1:n_node_feat
            upd_input[zero_off + k] = node_feat[node_off + k]
        end
        for k = 1:n_msg_feat
            upd_input[n_node_feat + k] = agg[agg_off + k]
        end
        for o = 1:n_node_feat
            s = b_upd[o]
            for i_seq_i = 1:n_in_upd
                widx = (o - 1) * n_in_upd + i_seq_i
                s = s + w_upd[widx] * upd_input[i_seq_i]
            end
            upd_scratch[o] = s
        end
        for k = 1:n_node_feat
            upd_scratch[k] = max(upd_scratch[k], 0.0)
        end
        for k = 1:n_node_feat
            node_feat_out[node_off + k] = upd_scratch[zero_off + k]
        end
    end
end
