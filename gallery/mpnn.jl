# zerofill(buf, n)
#
# Fills the first n elements of buf with zero.
#
# buf: array to clear, length at least n
# n: number of elements to clear
function zerofill(buf, n)
    for i = 1:n
        buf[i] = 0.0
    end
    return nothing
end

# mpnn_build_message(node_feat, edge_feat, src, dst, w_msg, b_msg,
#                     msg_input, msg_scratch, messages, e,
#                     n_node_feat, n_edge_feat, n_msg_feat, n_in_msg)
#
# Builds one edge's message: assembles [sender feature ; receiver feature ;
# edge feature], runs the message dense layer with a ReLU, and writes the
# result into this edge's own slice of messages.
#
# node_feat: node feature array, length n_nodes * n_node_feat, row-major
# edge_feat: edge feature array, length n_edges * n_edge_feat, row-major
# src: sender node index for each edge, length n_edges (1-based)
# dst: receiver node index for each edge, length n_edges (1-based)
# w_msg: message-layer weight, length n_msg_feat * n_in_msg
# b_msg: message-layer bias, length n_msg_feat
# msg_input: scratch array, length n_edges * n_in_msg -- this edge's own
#            private n_in_msg-slice is at offset (e-1)*n_in_msg
# msg_scratch: scratch array, length n_edges * n_msg_feat -- this edge's
#              own private n_msg_feat-slice is at offset (e-1)*n_msg_feat
# messages: output array, length n_edges * n_msg_feat, filled in place
# e: index of the edge this call builds a message for
# n_node_feat: number of node features
# n_edge_feat: number of edge features
# n_msg_feat: number of message features
# n_in_msg: width of the message layer's input, 2*n_node_feat+n_edge_feat
function mpnn_build_message(node_feat, edge_feat, src, dst, w_msg, b_msg, msg_input, msg_scratch, messages, e, n_node_feat, n_edge_feat, n_msg_feat, n_in_msg)
    s_node = src[e]
    d_node = dst[e]
    src_off = (s_node - 1) * n_node_feat
    dst_off = (d_node - 1) * n_node_feat
    edge_off = (e - 1) * n_edge_feat
    # this edge's own private slice of msg_input/msg_scratch -- required so
    # that GPU-splitting the caller's loop (one thread per edge) stays
    # race-free; a shared slice reused every call is only safe under
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
    return nothing
end

# mpnn_accumulate(messages, dst, agg, e, n_msg_feat)
#
# Adds one edge's message into its receiver node's running total. Two
# edges can share the same receiver, so a caller must run this
# sequentially over e to keep the shared updates to agg race-free.
#
# messages: message array, length n_edges * n_msg_feat, row-major
# dst: receiver node index for each edge, length n_edges (1-based)
# agg: per-node aggregate array, length n_nodes * n_msg_feat, updated
#      in place; must already be zero-filled before the first call
# e: index of the edge whose message this call adds in
# n_msg_feat: number of message features
function mpnn_accumulate(messages, dst, agg, e, n_msg_feat)
    d_node = dst[e]
    msg_off = (e - 1) * n_msg_feat
    agg_off = (d_node - 1) * n_msg_feat
    for j = 1:n_msg_feat
        agg[agg_off + j] = agg[agg_off + j] + messages[msg_off + j]
    end
    return nothing
end

# mpnn_update_node(node_feat, agg, w_upd, b_upd, upd_input, upd_scratch,
#                   node_feat_out, v, n_node_feat, n_msg_feat, n_in_upd)
#
# Updates one node's feature from its old value and its aggregated
# incoming messages: assembles [previous feature ; aggregate], runs the
# update dense layer with a ReLU, and writes the result for this node.
#
# node_feat: node feature array, length n_nodes * n_node_feat, row-major
# agg: per-node aggregate array, length n_nodes * n_msg_feat, row-major
# w_upd: update-layer weight, length n_node_feat * n_in_upd
# b_upd: update-layer bias, length n_node_feat
# upd_input: scratch array, length n_nodes * n_in_upd -- this node's own
#            private n_in_upd-slice is at offset (v-1)*n_in_upd
# upd_scratch: scratch array, length n_nodes * n_node_feat
# node_feat_out: output array, length n_nodes * n_node_feat, filled in place
# v: index of the node this call updates
# n_node_feat: number of node features
# n_msg_feat: number of message features
# n_in_upd: width of the update layer's input, n_node_feat+n_msg_feat
function mpnn_update_node(node_feat, agg, w_upd, b_upd, upd_input, upd_scratch, node_feat_out, v, n_node_feat, n_msg_feat, n_in_upd)
    node_off = (v - 1) * n_node_feat
    agg_off = (v - 1) * n_msg_feat
    # this node's own private slice of upd_input, for the same GPU-split-
    # race reason as mpnn_build_message's in_off; upd_scratch reuses
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
    return nothing
end

# mpnn(node_feat, edge_feat, src, dst, w_msg, b_msg, w_upd, b_upd,
#            n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat,
#            msg_input, msg_scratch, messages, agg, upd_input, upd_scratch,
#            node_feat_out)
#
# Runs one message-passing layer of a Message-Passing Neural Network:
# builds a message per edge, aggregates messages at each receiver node,
# then updates every node feature from its old value and its aggregate.
# Every stage below is its own kernel, called once per edge or per node
# from the loop that drives it.
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
# msg_input: scratch array, length n_edges * (2*n_node_feat + n_edge_feat)
# msg_scratch: scratch array, length n_edges * n_msg_feat
# messages: scratch array, length n_edges * n_msg_feat
# agg: scratch array, length n_nodes * n_msg_feat
# upd_input: scratch array, length n_nodes * (n_node_feat + n_msg_feat)
# upd_scratch: scratch array, length n_nodes * n_node_feat
# node_feat_out: output array, length n_nodes * n_node_feat, filled in place
function mpnn(node_feat, edge_feat, src, dst, w_msg, b_msg, w_upd, b_upd, n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat, msg_input, msg_scratch, messages, agg, upd_input, upd_scratch, node_feat_out)
    # sizes of the concatenated input vectors used by the two dense layers
    n_in_msg = 2 * n_node_feat + n_edge_feat
    n_in_upd = n_node_feat + n_msg_feat
    n_agg = n_nodes * n_msg_feat

    # clear the per-node aggregate before summing incoming messages
    zerofill(agg, n_agg)

    # step 1: compute one message per edge -- edges are independent of
    # each other, so this loop can run in any order
    for e = 1:n_edges
        mpnn_build_message(node_feat, edge_feat, src, dst, w_msg, b_msg, msg_input, msg_scratch, messages, e, n_node_feat, n_edge_feat, n_msg_feat, n_in_msg)
    end

    # step 2: sum messages at each receiver node -- sequential, since
    # different edges can share the same receiver node and every call
    # updates the shared agg array
    for i_e = 1:n_edges
        mpnn_accumulate(messages, dst, agg, i_e, n_msg_feat)
    end

    # step 3: update every node feature from its old value and its
    # aggregate -- nodes are independent of each other
    for v = 1:n_nodes
        mpnn_update_node(node_feat, agg, w_upd, b_upd, upd_input, upd_scratch, node_feat_out, v, n_node_feat, n_msg_feat, n_in_upd)
    end

    return nothing
end