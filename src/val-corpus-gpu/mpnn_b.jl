function initstacks_mpnn_b(n_edge_feat, n_edges, n_msg_feat, n_node_feat, n_nodes)
    msg_input_stack = Vector{Float64}(undef, ((div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1)) + (div(n_edges - 1, 1) + 1) * (div(n_edge_feat - 1, 1) + 1))
    msg_scratch_stack = Vector{Float64}(undef, (div(n_edges - 1, 1) + 1) * (div(n_msg_feat - 1, 1) + 1) + (div(n_edges - 1, 1) + 1) * (div(n_msg_feat - 1, 1) + 1))
    upd_input_stack = Vector{Float64}(undef, (div(n_nodes - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (div(n_nodes - 1, 1) + 1) * (div(n_msg_feat - 1, 1) + 1))
    upd_scratch_stack = Vector{Float64}(undef, (div(n_nodes - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (div(n_nodes - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1))
    return (msg_input_stack, msg_scratch_stack, upd_input_stack, upd_scratch_stack)
end

function mpnn_b(node_feat, node_featb, edge_feat, edge_featb, src, dst, w_msg, w_msgb, b_msg, b_msgb, w_upd, w_updb, b_upd, b_updb, n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat, msg_input, msg_inputb, msg_scratch, msg_scratchb, messages, messagesb, agg, aggb, upd_input, upd_inputb, upd_scratch, upd_scratchb, node_feat_out, node_feat_outb, msg_input_stack, msg_scratch_stack, upd_input_stack, upd_scratch_stack)
    s = 0.0
    sb = 0.0
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
        in_off = (e - 1) * n_in_msg
        msg_off = (e - 1) * n_msg_feat
        for k = 1:n_node_feat
            msg_input_stack[((e - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1] = msg_input[in_off + k]
            msg_input[in_off + k] = node_feat[src_off + k]
        end
        for k = 1:n_node_feat
            msg_input_stack[(div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (((e - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1)] = msg_input[in_off + n_node_feat + k]
            msg_input[in_off + n_node_feat + k] = node_feat[dst_off + k]
        end
        for k = 1:n_edge_feat
            msg_input_stack[((div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1)) + (((e - 1) * (div(n_edge_feat - 1, 1) + 1) + (k - 1)) + 1)] = msg_input[in_off + 2n_node_feat + k]
            msg_input[in_off + 2n_node_feat + k] = edge_feat[edge_off + k]
        end
        for o = 1:n_msg_feat
            s = b_msg[o]
            for i_seq_i = 1:n_in_msg
                widx = (o - 1) * n_in_msg + i_seq_i
                s = s + w_msg[widx] * msg_input[in_off + i_seq_i]
            end
            msg_scratch_stack[((e - 1) * (div(n_msg_feat - 1, 1) + 1) + (o - 1)) + 1] = msg_scratch[msg_off + o]
            msg_scratch[msg_off + o] = s
        end
        for k = 1:n_msg_feat
            msg_scratch_stack[(div(n_edges - 1, 1) + 1) * (div(n_msg_feat - 1, 1) + 1) + (((e - 1) * (div(n_msg_feat - 1, 1) + 1) + (k - 1)) + 1)] = msg_scratch[msg_off + k]
            msg_scratch[msg_off + k] = max(msg_scratch[msg_off + k], 0.0)
        end
        for k = 1:n_msg_feat
            messages[msg_off + k] = msg_scratch[msg_off + k]
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
        uin_off = (v - 1) * n_in_upd
        for k = 1:n_node_feat
            upd_input_stack[((v - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1] = upd_input[uin_off + k]
            upd_input[uin_off + k] = node_feat[node_off + k]
        end
        for k = 1:n_msg_feat
            upd_input_stack[(div(n_nodes - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (((v - 1) * (div(n_msg_feat - 1, 1) + 1) + (k - 1)) + 1)] = upd_input[uin_off + n_node_feat + k]
            upd_input[uin_off + n_node_feat + k] = agg[agg_off + k]
        end
        for o = 1:n_node_feat
            s = b_upd[o]
            for i_seq_i = 1:n_in_upd
                widx = (o - 1) * n_in_upd + i_seq_i
                s = s + w_upd[widx] * upd_input[uin_off + i_seq_i]
            end
            upd_scratch_stack[((v - 1) * (div(n_node_feat - 1, 1) + 1) + (o - 1)) + 1] = upd_scratch[node_off + o]
            upd_scratch[node_off + o] = s
        end
        for k = 1:n_node_feat
            upd_scratch_stack[(div(n_nodes - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (((v - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1)] = upd_scratch[node_off + k]
            upd_scratch[node_off + k] = max(upd_scratch[node_off + k], 0.0)
        end
        for k = 1:n_node_feat
            node_feat_out[node_off + k] = upd_scratch[node_off + k]
        end
    end
    n_in_msg = 2n_node_feat + n_edge_feat
    n_in_upd = n_node_feat + n_msg_feat
    n_agg = n_nodes * n_msg_feat
    for v = n_nodes:-1:1
        node_off = (v - 1) * n_node_feat
        agg_off = (v - 1) * n_msg_feat
        uin_off = (v - 1) * n_in_upd
        for k = 1:n_node_feat
            upd_scratchb[node_off + k] = upd_scratchb[node_off + k] + node_feat_outb[node_off + k]
            node_feat_outb[node_off + k] = 0.0
        end
        for k = n_node_feat:-1:1
            upd_scratch[node_off + k] = upd_scratch_stack[(div(n_nodes - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (((v - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1)]
            upd_scratchb[node_off + k] = (0.5 * (1.0 + sign(upd_scratch[node_off + k]))) * upd_scratchb[node_off + k]
        end
        for o = n_node_feat:-1:1
            upd_scratch[node_off + o] = upd_scratch_stack[((v - 1) * (div(n_node_feat - 1, 1) + 1) + (o - 1)) + 1]
            sb = sb + upd_scratchb[node_off + o]
            upd_scratchb[node_off + o] = 0.0
            for i_seq_i = n_in_upd:-1:1
                widx = (o - 1) * n_in_upd + i_seq_i
                w_updb[widx] = w_updb[widx] + upd_input[uin_off + i_seq_i] * sb
                upd_inputb[uin_off + i_seq_i] = upd_inputb[uin_off + i_seq_i] + w_upd[widx] * sb
            end
            b_updb[o] = b_updb[o] + sb
            sb = 0.0
        end
        for k = n_msg_feat:-1:1
            upd_input[uin_off + n_node_feat + k] = upd_input_stack[(div(n_nodes - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (((v - 1) * (div(n_msg_feat - 1, 1) + 1) + (k - 1)) + 1)]
            aggb[agg_off + k] = aggb[agg_off + k] + upd_inputb[uin_off + n_node_feat + k]
            upd_inputb[uin_off + n_node_feat + k] = 0.0
        end
        for k = n_node_feat:-1:1
            upd_input[uin_off + k] = upd_input_stack[((v - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1]
            node_featb[node_off + k] = node_featb[node_off + k] + upd_inputb[uin_off + k]
            upd_inputb[uin_off + k] = 0.0
        end
    end
    for i_seq_e = n_edges:-1:1
        d_node = dst[i_seq_e]
        msg_off = (i_seq_e - 1) * n_msg_feat
        agg_off = (d_node - 1) * n_msg_feat
        for j = 1:n_msg_feat
            messagesb[msg_off + j] = messagesb[msg_off + j] + aggb[agg_off + j]
        end
    end
    for e = n_edges:-1:1
        s_node = src[e]
        d_node = dst[e]
        src_off = (s_node - 1) * n_node_feat
        dst_off = (d_node - 1) * n_node_feat
        edge_off = (e - 1) * n_edge_feat
        in_off = (e - 1) * n_in_msg
        msg_off = (e - 1) * n_msg_feat
        for k = 1:n_msg_feat
            msg_scratchb[msg_off + k] = msg_scratchb[msg_off + k] + messagesb[msg_off + k]
            messagesb[msg_off + k] = 0.0
        end
        for k = n_msg_feat:-1:1
            msg_scratch[msg_off + k] = msg_scratch_stack[(div(n_edges - 1, 1) + 1) * (div(n_msg_feat - 1, 1) + 1) + (((e - 1) * (div(n_msg_feat - 1, 1) + 1) + (k - 1)) + 1)]
            msg_scratchb[msg_off + k] = (0.5 * (1.0 + sign(msg_scratch[msg_off + k]))) * msg_scratchb[msg_off + k]
        end
        for o = n_msg_feat:-1:1
            msg_scratch[msg_off + o] = msg_scratch_stack[((e - 1) * (div(n_msg_feat - 1, 1) + 1) + (o - 1)) + 1]
            sb = sb + msg_scratchb[msg_off + o]
            msg_scratchb[msg_off + o] = 0.0
            for i_seq_i = n_in_msg:-1:1
                widx = (o - 1) * n_in_msg + i_seq_i
                w_msgb[widx] = w_msgb[widx] + msg_input[in_off + i_seq_i] * sb
                msg_inputb[in_off + i_seq_i] = msg_inputb[in_off + i_seq_i] + w_msg[widx] * sb
            end
            b_msgb[o] = b_msgb[o] + sb
            sb = 0.0
        end
        for k = n_edge_feat:-1:1
            msg_input[in_off + 2n_node_feat + k] = msg_input_stack[((div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1)) + (((e - 1) * (div(n_edge_feat - 1, 1) + 1) + (k - 1)) + 1)]
            edge_featb[edge_off + k] = edge_featb[edge_off + k] + msg_inputb[in_off + 2n_node_feat + k]
            msg_inputb[in_off + 2n_node_feat + k] = 0.0
        end
        for k = n_node_feat:-1:1
            msg_input[in_off + n_node_feat + k] = msg_input_stack[(div(n_edges - 1, 1) + 1) * (div(n_node_feat - 1, 1) + 1) + (((e - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1)]
            node_featb[dst_off + k] = node_featb[dst_off + k] + msg_inputb[in_off + n_node_feat + k]
            msg_inputb[in_off + n_node_feat + k] = 0.0
        end
        for k = n_node_feat:-1:1
            msg_input[in_off + k] = msg_input_stack[((e - 1) * (div(n_node_feat - 1, 1) + 1) + (k - 1)) + 1]
            node_featb[src_off + k] = node_featb[src_off + k] + msg_inputb[in_off + k]
            msg_inputb[in_off + k] = 0.0
        end
    end
    for k = 1:n_agg
        aggb[k] = 0.0
    end
    return nothing
end

function mpnn(node_feat, edge_feat, src, dst, w_msg, b_msg, w_upd, b_upd, n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat, msg_input, msg_scratch, messages, agg, upd_input, upd_scratch, node_feat_out)
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
    end
end
