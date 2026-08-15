function initstacks_mpnn_b()
    msg_input_stack = Vector{Float64}()
    msg_scratch_stack = Vector{Float64}()
    upd_input_stack = Vector{Float64}()
    upd_scratch_stack = Vector{Float64}()
    return (msg_input_stack, msg_scratch_stack, upd_input_stack, upd_scratch_stack)
end

function mpnn_hv(node_feat, node_featb, edge_feat, edge_featb, src, dst, w_msg, w_msgb, b_msg, b_msgb, w_upd, w_updb, b_upd, b_updb, n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat, msg_input, msg_inputb, msg_scratch, msg_scratchb, messages, messagesb, agg, aggb, upd_input, upd_inputb, upd_scratch, upd_scratchb, node_feat_out, node_feat_outb, node_featd, node_featbd, edge_featd, edge_featbd, w_msgd, w_msgbd, b_msgd, b_msgbd, w_updd, w_updbd, b_updd, b_updbd, msg_inputd, msg_inputbd, msg_scratchd, msg_scratchbd, messagesd, messagesbd, aggd, aggbd, upd_inputd, upd_inputbd, upd_scratchd, upd_scratchbd, node_feat_outd, node_feat_outbd, msg_input_stack, msg_scratch_stack, upd_input_stack, upd_scratch_stack)
    msg_input_stack_d = Vector{Float64}()
    msg_scratch_stack_d = Vector{Float64}()
    upd_input_stack_d = Vector{Float64}()
    upd_scratch_stack_d = Vector{Float64}()
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    zero_off = 0
    n_in_msg = 2n_node_feat + n_edge_feat
    n_in_upd = n_node_feat + n_msg_feat
    n_agg = n_nodes * n_msg_feat
    for k = 1:n_agg
        aggd[k] = 0.0
        agg[k] = 0.0
    end
    for e = 1:n_edges
        s_node = src[e]
        d_node = dst[e]
        src_off = (s_node - 1) * n_node_feat
        dst_off = (d_node - 1) * n_node_feat
        edge_off = (e - 1) * n_edge_feat
        for k = 1:n_node_feat
            push!(msg_input_stack_d, msg_inputd[zero_off + k])
            push!(msg_input_stack, msg_input[zero_off + k])
            msg_inputd[zero_off + k] = node_featd[src_off + k]
            msg_input[zero_off + k] = node_feat[src_off + k]
        end
        for k = 1:n_node_feat
            push!(msg_input_stack_d, msg_inputd[n_node_feat + k])
            push!(msg_input_stack, msg_input[n_node_feat + k])
            msg_inputd[n_node_feat + k] = node_featd[dst_off + k]
            msg_input[n_node_feat + k] = node_feat[dst_off + k]
        end
        for k = 1:n_edge_feat
            push!(msg_input_stack_d, msg_inputd[2n_node_feat + k])
            push!(msg_input_stack, msg_input[2n_node_feat + k])
            msg_inputd[2n_node_feat + k] = edge_featd[edge_off + k]
            msg_input[2n_node_feat + k] = edge_feat[edge_off + k]
        end
        for o = 1:n_msg_feat
            sd = b_msgd[o]
            s = b_msg[o]
            for i_seq_i = 1:n_in_msg
                widx = (o - 1) * n_in_msg + i_seq_i
                sd = sd + (msg_input[i_seq_i] * w_msgd[widx] + w_msg[widx] * msg_inputd[i_seq_i])
                s = s + w_msg[widx] * msg_input[i_seq_i]
            end
            push!(msg_scratch_stack_d, msg_scratchd[o])
            push!(msg_scratch_stack, msg_scratch[o])
            msg_scratchd[o] = sd
            msg_scratch[o] = s
        end
        for k = 1:n_msg_feat
            push!(msg_scratch_stack_d, msg_scratchd[k])
            push!(msg_scratch_stack, msg_scratch[k])
            msg_scratchd[k] = (0.5 * (1.0 + sign(msg_scratch[k]))) * msg_scratchd[k]
            msg_scratch[k] = max(msg_scratch[k], 0.0)
        end
        msg_off = (e - 1) * n_msg_feat
        for k = 1:n_msg_feat
            messagesd[msg_off + k] = msg_scratchd[zero_off + k]
            messages[msg_off + k] = msg_scratch[zero_off + k]
        end
    end
    for i_seq_e = 1:n_edges
        d_node = dst[i_seq_e]
        msg_off = (i_seq_e - 1) * n_msg_feat
        agg_off = (d_node - 1) * n_msg_feat
        for j = 1:n_msg_feat
            aggd[agg_off + j] = aggd[agg_off + j] + messagesd[msg_off + j]
            agg[agg_off + j] = agg[agg_off + j] + messages[msg_off + j]
        end
    end
    for v = 1:n_nodes
        node_off = (v - 1) * n_node_feat
        agg_off = (v - 1) * n_msg_feat
        for k = 1:n_node_feat
            push!(upd_input_stack_d, upd_inputd[zero_off + k])
            push!(upd_input_stack, upd_input[zero_off + k])
            upd_inputd[zero_off + k] = node_featd[node_off + k]
            upd_input[zero_off + k] = node_feat[node_off + k]
        end
        for k = 1:n_msg_feat
            push!(upd_input_stack_d, upd_inputd[n_node_feat + k])
            push!(upd_input_stack, upd_input[n_node_feat + k])
            upd_inputd[n_node_feat + k] = aggd[agg_off + k]
            upd_input[n_node_feat + k] = agg[agg_off + k]
        end
        for o = 1:n_node_feat
            sd = b_updd[o]
            s = b_upd[o]
            for i_seq_i = 1:n_in_upd
                widx = (o - 1) * n_in_upd + i_seq_i
                sd = sd + (upd_input[i_seq_i] * w_updd[widx] + w_upd[widx] * upd_inputd[i_seq_i])
                s = s + w_upd[widx] * upd_input[i_seq_i]
            end
            push!(upd_scratch_stack_d, upd_scratchd[o])
            push!(upd_scratch_stack, upd_scratch[o])
            upd_scratchd[o] = sd
            upd_scratch[o] = s
        end
        for k = 1:n_node_feat
            push!(upd_scratch_stack_d, upd_scratchd[k])
            push!(upd_scratch_stack, upd_scratch[k])
            upd_scratchd[k] = (0.5 * (1.0 + sign(upd_scratch[k]))) * upd_scratchd[k]
            upd_scratch[k] = max(upd_scratch[k], 0.0)
        end
        for k = 1:n_node_feat
            node_feat_outd[node_off + k] = upd_scratchd[zero_off + k]
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
            upd_scratchbd[zero_off + k] = upd_scratchbd[zero_off + k] + node_feat_outbd[node_off + k]
            upd_scratchb[zero_off + k] = upd_scratchb[zero_off + k] + node_feat_outb[node_off + k]
            node_feat_outbd[node_off + k] = 0.0
            node_feat_outb[node_off + k] = 0.0
        end
        for k = n_node_feat:-1:1
            upd_scratchd[k] = pop!(upd_scratch_stack_d)
            upd_scratch[k] = pop!(upd_scratch_stack)
            upd_scratchbd[k] = (0.5 * (1.0 + sign(upd_scratch[k]))) * upd_scratchbd[k]
            upd_scratchb[k] = (0.5 * (1.0 + sign(upd_scratch[k]))) * upd_scratchb[k]
        end
        for o = n_node_feat:-1:1
            upd_scratchd[o] = pop!(upd_scratch_stack_d)
            upd_scratch[o] = pop!(upd_scratch_stack)
            sbd = sbd + upd_scratchbd[o]
            sb = sb + upd_scratchb[o]
            upd_scratchbd[o] = 0.0
            upd_scratchb[o] = 0.0
            for i_seq_i = n_in_upd:-1:1
                widx = (o - 1) * n_in_upd + i_seq_i
                w_updbd[widx] = w_updbd[widx] + (sb * upd_inputd[i_seq_i] + upd_input[i_seq_i] * sbd)
                w_updb[widx] = w_updb[widx] + upd_input[i_seq_i] * sb
                upd_inputbd[i_seq_i] = upd_inputbd[i_seq_i] + (sb * w_updd[widx] + w_upd[widx] * sbd)
                upd_inputb[i_seq_i] = upd_inputb[i_seq_i] + w_upd[widx] * sb
            end
            b_updbd[o] = b_updbd[o] + sbd
            b_updb[o] = b_updb[o] + sb
            sbd = 0.0
            sb = 0.0
        end
        for k = n_msg_feat:-1:1
            upd_inputd[n_node_feat + k] = pop!(upd_input_stack_d)
            upd_input[n_node_feat + k] = pop!(upd_input_stack)
            aggbd[agg_off + k] = aggbd[agg_off + k] + upd_inputbd[n_node_feat + k]
            aggb[agg_off + k] = aggb[agg_off + k] + upd_inputb[n_node_feat + k]
            upd_inputbd[n_node_feat + k] = 0.0
            upd_inputb[n_node_feat + k] = 0.0
        end
        for k = n_node_feat:-1:1
            upd_inputd[zero_off + k] = pop!(upd_input_stack_d)
            upd_input[zero_off + k] = pop!(upd_input_stack)
            node_featbd[node_off + k] = node_featbd[node_off + k] + upd_inputbd[zero_off + k]
            node_featb[node_off + k] = node_featb[node_off + k] + upd_inputb[zero_off + k]
            upd_inputbd[zero_off + k] = 0.0
            upd_inputb[zero_off + k] = 0.0
        end
    end
    for i_seq_e = n_edges:-1:1
        d_node = dst[i_seq_e]
        msg_off = (i_seq_e - 1) * n_msg_feat
        agg_off = (d_node - 1) * n_msg_feat
        for j = 1:n_msg_feat
            messagesbd[msg_off + j] = messagesbd[msg_off + j] + aggbd[agg_off + j]
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
            msg_scratchbd[zero_off + k] = msg_scratchbd[zero_off + k] + messagesbd[msg_off + k]
            msg_scratchb[zero_off + k] = msg_scratchb[zero_off + k] + messagesb[msg_off + k]
            messagesbd[msg_off + k] = 0.0
            messagesb[msg_off + k] = 0.0
        end
        for k = n_msg_feat:-1:1
            msg_scratchd[k] = pop!(msg_scratch_stack_d)
            msg_scratch[k] = pop!(msg_scratch_stack)
            msg_scratchbd[k] = (0.5 * (1.0 + sign(msg_scratch[k]))) * msg_scratchbd[k]
            msg_scratchb[k] = (0.5 * (1.0 + sign(msg_scratch[k]))) * msg_scratchb[k]
        end
        for o = n_msg_feat:-1:1
            msg_scratchd[o] = pop!(msg_scratch_stack_d)
            msg_scratch[o] = pop!(msg_scratch_stack)
            sbd = sbd + msg_scratchbd[o]
            sb = sb + msg_scratchb[o]
            msg_scratchbd[o] = 0.0
            msg_scratchb[o] = 0.0
            for i_seq_i = n_in_msg:-1:1
                widx = (o - 1) * n_in_msg + i_seq_i
                w_msgbd[widx] = w_msgbd[widx] + (sb * msg_inputd[i_seq_i] + msg_input[i_seq_i] * sbd)
                w_msgb[widx] = w_msgb[widx] + msg_input[i_seq_i] * sb
                msg_inputbd[i_seq_i] = msg_inputbd[i_seq_i] + (sb * w_msgd[widx] + w_msg[widx] * sbd)
                msg_inputb[i_seq_i] = msg_inputb[i_seq_i] + w_msg[widx] * sb
            end
            b_msgbd[o] = b_msgbd[o] + sbd
            b_msgb[o] = b_msgb[o] + sb
            sbd = 0.0
            sb = 0.0
        end
        for k = n_edge_feat:-1:1
            msg_inputd[2n_node_feat + k] = pop!(msg_input_stack_d)
            msg_input[2n_node_feat + k] = pop!(msg_input_stack)
            edge_featbd[edge_off + k] = edge_featbd[edge_off + k] + msg_inputbd[2n_node_feat + k]
            edge_featb[edge_off + k] = edge_featb[edge_off + k] + msg_inputb[2n_node_feat + k]
            msg_inputbd[2n_node_feat + k] = 0.0
            msg_inputb[2n_node_feat + k] = 0.0
        end
        for k = n_node_feat:-1:1
            msg_inputd[n_node_feat + k] = pop!(msg_input_stack_d)
            msg_input[n_node_feat + k] = pop!(msg_input_stack)
            node_featbd[dst_off + k] = node_featbd[dst_off + k] + msg_inputbd[n_node_feat + k]
            node_featb[dst_off + k] = node_featb[dst_off + k] + msg_inputb[n_node_feat + k]
            msg_inputbd[n_node_feat + k] = 0.0
            msg_inputb[n_node_feat + k] = 0.0
        end
        for k = n_node_feat:-1:1
            msg_inputd[zero_off + k] = pop!(msg_input_stack_d)
            msg_input[zero_off + k] = pop!(msg_input_stack)
            node_featbd[src_off + k] = node_featbd[src_off + k] + msg_inputbd[zero_off + k]
            node_featb[src_off + k] = node_featb[src_off + k] + msg_inputb[zero_off + k]
            msg_inputbd[zero_off + k] = 0.0
            msg_inputb[zero_off + k] = 0.0
        end
    end
    for k = 1:n_agg
        aggbd[k] = 0.0
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
