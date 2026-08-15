function initstacks_mpnn_b()
    msg_input_stack = Vector{Float64}()
    msg_scratch_stack = Vector{Float64}()
    upd_input_stack = Vector{Float64}()
    upd_scratch_stack = Vector{Float64}()
    return (msg_input_stack, msg_scratch_stack, upd_input_stack, upd_scratch_stack)
end

function mpnn_b(node_feat, node_featb, edge_feat, edge_featb, src, dst, w_msg, w_msgb, b_msg, b_msgb, w_upd, w_updb, b_upd, b_updb, n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat, msg_input, msg_inputb, msg_scratch, msg_scratchb, messages, messagesb, agg, aggb, upd_input, upd_inputb, upd_scratch, upd_scratchb, node_feat_out, node_feat_outb, msg_input_stack, msg_scratch_stack, upd_input_stack, upd_scratch_stack)
    s = 0.0
    sb = 0.0
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
            push!(msg_input_stack, msg_input[zero_off + k])
            msg_input[zero_off + k] = node_feat[src_off + k]
        end
        for k = 1:n_node_feat
            push!(msg_input_stack, msg_input[n_node_feat + k])
            msg_input[n_node_feat + k] = node_feat[dst_off + k]
        end
        for k = 1:n_edge_feat
            push!(msg_input_stack, msg_input[2n_node_feat + k])
            msg_input[2n_node_feat + k] = edge_feat[edge_off + k]
        end
        for o = 1:n_msg_feat
            s = b_msg[o]
            for i_seq_i = 1:n_in_msg
                widx = (o - 1) * n_in_msg + i_seq_i
                s = s + w_msg[widx] * msg_input[i_seq_i]
            end
            push!(msg_scratch_stack, msg_scratch[o])
            msg_scratch[o] = s
        end
        for k = 1:n_msg_feat
            push!(msg_scratch_stack, msg_scratch[k])
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
            push!(upd_input_stack, upd_input[zero_off + k])
            upd_input[zero_off + k] = node_feat[node_off + k]
        end
        for k = 1:n_msg_feat
            push!(upd_input_stack, upd_input[n_node_feat + k])
            upd_input[n_node_feat + k] = agg[agg_off + k]
        end
        for o = 1:n_node_feat
            s = b_upd[o]
            for i_seq_i = 1:n_in_upd
                widx = (o - 1) * n_in_upd + i_seq_i
                s = s + w_upd[widx] * upd_input[i_seq_i]
            end
            push!(upd_scratch_stack, upd_scratch[o])
            upd_scratch[o] = s
        end
        for k = 1:n_node_feat
            push!(upd_scratch_stack, upd_scratch[k])
            upd_scratch[k] = max(upd_scratch[k], 0.0)
        end
        for k = 1:n_node_feat
            node_feat_out[node_off + k] = upd_scratch[zero_off + k]
        end
    end
    zero_off = 0
    n_in_msg = 2n_node_feat + n_edge_feat
    n_in_upd = n_node_feat + n_msg_feat
    n_agg = n_nodes * n_msg_feat
    for v = n_nodes:-1:1
        node_off = (v - 1) * n_node_feat
        agg_off = (v - 1) * n_msg_feat
        for k = 1:n_node_feat
            upd_scratchb[zero_off + k] = upd_scratchb[zero_off + k] + node_feat_outb[node_off + k]
            node_feat_outb[node_off + k] = 0.0
        end
        for k = n_node_feat:-1:1
            upd_scratch[k] = pop!(upd_scratch_stack)
            upd_scratchb[k] = (0.5 * (1.0 + sign(upd_scratch[k]))) * upd_scratchb[k]
        end
        for o = n_node_feat:-1:1
            upd_scratch[o] = pop!(upd_scratch_stack)
            sb = sb + upd_scratchb[o]
            upd_scratchb[o] = 0.0
            for i_seq_i = n_in_upd:-1:1
                widx = (o - 1) * n_in_upd + i_seq_i
                w_updb[widx] = w_updb[widx] + upd_input[i_seq_i] * sb
                upd_inputb[i_seq_i] = upd_inputb[i_seq_i] + w_upd[widx] * sb
            end
            b_updb[o] = b_updb[o] + sb
            sb = 0.0
        end
        for k = n_msg_feat:-1:1
            upd_input[n_node_feat + k] = pop!(upd_input_stack)
            aggb[agg_off + k] = aggb[agg_off + k] + upd_inputb[n_node_feat + k]
            upd_inputb[n_node_feat + k] = 0.0
        end
        for k = n_node_feat:-1:1
            upd_input[zero_off + k] = pop!(upd_input_stack)
            node_featb[node_off + k] = node_featb[node_off + k] + upd_inputb[zero_off + k]
            upd_inputb[zero_off + k] = 0.0
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
        msg_off = (e - 1) * n_msg_feat
        for k = 1:n_msg_feat
            msg_scratchb[zero_off + k] = msg_scratchb[zero_off + k] + messagesb[msg_off + k]
            messagesb[msg_off + k] = 0.0
        end
        for k = n_msg_feat:-1:1
            msg_scratch[k] = pop!(msg_scratch_stack)
            msg_scratchb[k] = (0.5 * (1.0 + sign(msg_scratch[k]))) * msg_scratchb[k]
        end
        for o = n_msg_feat:-1:1
            msg_scratch[o] = pop!(msg_scratch_stack)
            sb = sb + msg_scratchb[o]
            msg_scratchb[o] = 0.0
            for i_seq_i = n_in_msg:-1:1
                widx = (o - 1) * n_in_msg + i_seq_i
                w_msgb[widx] = w_msgb[widx] + msg_input[i_seq_i] * sb
                msg_inputb[i_seq_i] = msg_inputb[i_seq_i] + w_msg[widx] * sb
            end
            b_msgb[o] = b_msgb[o] + sb
            sb = 0.0
        end
        for k = n_edge_feat:-1:1
            msg_input[2n_node_feat + k] = pop!(msg_input_stack)
            edge_featb[edge_off + k] = edge_featb[edge_off + k] + msg_inputb[2n_node_feat + k]
            msg_inputb[2n_node_feat + k] = 0.0
        end
        for k = n_node_feat:-1:1
            msg_input[n_node_feat + k] = pop!(msg_input_stack)
            node_featb[dst_off + k] = node_featb[dst_off + k] + msg_inputb[n_node_feat + k]
            msg_inputb[n_node_feat + k] = 0.0
        end
        for k = n_node_feat:-1:1
            msg_input[zero_off + k] = pop!(msg_input_stack)
            node_featb[src_off + k] = node_featb[src_off + k] + msg_inputb[zero_off + k]
            msg_inputb[zero_off + k] = 0.0
        end
    end
    for k = 1:n_agg
        aggb[k] = 0.0
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
