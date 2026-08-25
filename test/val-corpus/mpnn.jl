# mpnn(node_feat, edge_feat, src, dst, w_msg, b_msg, w_upd, b_upd,
#             n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat,
#             msg_input, msg_scratch, messages, agg, upd_input, upd_scratch,
#             node_feat_out)
#
# Runs one message-passing layer of a Message-Passing Neural Network in a
# single function: builds a message per edge, aggregates messages at each
# receiver node, and updates every node feature from its old value and its
# aggregate.
#
# node_feat: node feature array, length n_nodes * n_node_feat, row-major
# edge_feat: edge feature array, length n_edges * n_edge_feat, row-major
# src: sender node index for each edge, length n_edges (1-based)
# dst: receiver node index for each edge, length n_edges (1-based)
# w_msg: message-layer weight, length n_msg_feat * (2*n_node_feat+n_edge_feat)
# b_msg: message-layer bias, length n_msg_feat
# w_upd: update-layer weight, length n_node_feat * (n_node_feat + n_msg_feat)
# b_upd: update-layer bias, length n_node_feat
# n_nodes: number of nodes
# n_edges: number of edges
# n_node_feat: number of node features
# n_edge_feat: number of edge features
# n_msg_feat: number of message features
# msg_input: scratch array, length n_edges * (2*n_node_feat + n_edge_feat) --
#            one private (2*n_node_feat + n_edge_feat)-slice per edge
# msg_scratch: scratch array, length n_edges * n_msg_feat --
#              one private n_msg_feat-slice per edge
# messages: scratch array, length n_edges * n_msg_feat
# agg: scratch array, length n_nodes * n_msg_feat
# upd_input: scratch array, length n_nodes * (n_node_feat + n_msg_feat) --
#            one private (n_node_feat + n_msg_feat)-slice per node
# upd_scratch: scratch array, length n_nodes * n_node_feat --
#              one private n_node_feat-slice per node
# node_feat_out: output array, length n_nodes * n_node_feat, filled in place
function mpnn(node_feat, edge_feat, src, dst, w_msg, b_msg, w_upd, b_upd, n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat, msg_input, msg_scratch, messages, agg, upd_input, upd_scratch, node_feat_out)
    # named zero offset, used wherever a copy starts at the beginning of an array
    # zero_off = 0
    # sizes of the concatenated input vectors used by the two dense layers
    n_in_msg = 2 * n_node_feat + n_edge_feat
    n_in_upd = n_node_feat + n_msg_feat
    n_agg = n_nodes * n_msg_feat

    # clear the per-node aggregate before summing incoming messages
    for k = 1:n_agg
        agg[k] = 0.0
    end

    # step 1: compute one message per edge
    for e = 1:n_edges
        s_node = src[e]
        d_node = dst[e]
        src_off = (s_node - 1) * n_node_feat
        dst_off = (d_node - 1) * n_node_feat
        edge_off = (e - 1) * n_edge_feat
        # this edge's own private slice of msg_input/msg_scratch -- required
        # so that GPU-splitting this loop (one thread per edge) is race-free;
        # a single shared slice reused every iteration is only safe under
        # strictly sequential (CPU) execution
        in_off = (e - 1) * n_in_msg
        msg_off = (e - 1) * n_msg_feat
        # assemble [sender feature ; receiver feature ; edge feature]
        for k = 1:n_node_feat
            msg_input[in_off + k] = node_feat[src_off + k]
        end
        for k = 1:n_node_feat
            msg_input[in_off + n_node_feat + k] = node_feat[dst_off + k]
        end
        for k = 1:n_edge_feat
            msg_input[in_off + 2 * n_node_feat + k] = edge_feat[edge_off + k]
        end
        # dense layer: msg_scratch = w_msg * msg_input + b_msg
        for o = 1:n_msg_feat
            s = b_msg[o]
            # accumulate the dot product sequentially over the input features
            for i_i = 1:n_in_msg
                widx = (o - 1) * n_in_msg + i_i
                s = s + w_msg[widx] * msg_input[in_off + i_i]
            end
            msg_scratch[msg_off + o] = s
        end
        # ReLU activation
        for k = 1:n_msg_feat
            msg_scratch[msg_off + k] = max(msg_scratch[msg_off + k], 0.0)
        end
        for k = 1:n_msg_feat
            messages[msg_off + k] = msg_scratch[msg_off + k]
        end
    end

    # step 2: sum messages at each receiver node
    # sequential: different edges can share the same receiver node, so
    # this loop carries a dependency through the shared agg array
    for i_e = 1:n_edges
        d_node = dst[i_e]
        msg_off = (i_e - 1) * n_msg_feat
        agg_off = (d_node - 1) * n_msg_feat
        # add this edge's message into its receiver's running total
        for j = 1:n_msg_feat
            agg[agg_off + j] = agg[agg_off + j] + messages[msg_off + j]
        end
    end

    # step 3: update every node feature from its old value and its aggregate
    for v = 1:n_nodes
        node_off = (v - 1) * n_node_feat
        agg_off = (v - 1) * n_msg_feat
        # this node's own private slice of upd_input, for the same
        # GPU-split-race reason as in_off/msg_off above; upd_scratch reuses
        # node_off directly since both it and node_feat_out are already
        # n_node_feat-per-node
        uin_off = (v - 1) * n_in_upd
        # assemble [previous feature ; aggregated incoming message]
        for k = 1:n_node_feat
            upd_input[uin_off + k] = node_feat[node_off + k]
        end
        for k = 1:n_msg_feat
            upd_input[uin_off + n_node_feat + k] = agg[agg_off + k]
        end
        # dense layer: upd_scratch = w_upd * upd_input + b_upd
        for o = 1:n_node_feat
            s = b_upd[o]
            # accumulate the dot product sequentially over the input features
            for i_i = 1:n_in_upd
                widx = (o - 1) * n_in_upd + i_i
                s = s + w_upd[widx] * upd_input[uin_off + i_i]
            end
            upd_scratch[node_off + o] = s
        end
        # ReLU activation
        for k = 1:n_node_feat
            upd_scratch[node_off + k] = max(upd_scratch[node_off + k], 0.0)
        end
        for k = 1:n_node_feat
            node_feat_out[node_off + k] = upd_scratch[node_off + k]
        end
    end

    return nothing
end