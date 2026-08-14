# copy_segment(src, src_off, dst, dst_off, n)
#
# Copies n consecutive entries from src (starting right after src_off)
# into dst (starting right after dst_off). Used to assemble concatenated
# feature vectors without indirect indexing.
#
# src: source array
# src_off: offset into src; element (src_off + k) is copied for k = 1:n
# dst: destination array, filled in place
# dst_off: offset into dst; element (dst_off + k) receives the copy
# n: number of entries to copy
function copy_segment(src, src_off, dst, dst_off, n)
    # each copied entry only depends on its own source entry
    for k = 1:n
        dst[dst_off + k] = src[src_off + k]
    end
    return nothing
end

# zero_vec(x, n)
#
# Sets the first n entries of x to 0.0.
#
# x: array to clear, filled in place
# n: number of leading entries to zero
function zero_vec(x, n)
    # each entry is cleared independently of the others
    for k = 1:n
        x[k] = 0.0
    end
    return nothing
end

# relu_vec(x, n, y)
#
# Elementwise rectified-linear activation: y[k] = max(x[k], 0.0).
#
# x: input array of length n
# n: number of entries
# y: output array of length n, filled in place (may alias x)
function relu_vec(x, n, y)
    # each output entry only depends on the matching input entry
    for k = 1:n
        y[k] = max(x[k], 0.0)
    end
    return nothing
end

# linear_transform(w, b, x, n_in, n_out, y)
#
# Dense layer: y = w * x + b, with w stored row-major (output o, input i
# at (o - 1) * n_in + i).
#
# w: weight array of length n_out * n_in, row-major
# b: bias array of length n_out
# x: input array of length n_in
# n_in: number of input features
# n_out: number of output features
# y: output array of length n_out, filled in place
function linear_transform(w, b, x, n_in, n_out, y)
    # each output feature is independent of the others
    for o = 1:n_out
        s = b[o]
        # accumulate the dot product sequentially over the input features
        for i_seq_i = 1:n_in
            widx = (o - 1) * n_in + i_seq_i
            s = s + w[widx] * x[i_seq_i]
        end
        y[o] = s
    end
    return nothing
end

# message_function(node_feat, edge_feat, src, dst, w_msg, b_msg, n_edges,
#                   n_node_feat, n_edge_feat, n_msg_feat, msg_input,
#                   msg_scratch, messages)
#
# Computes one message per edge from the sender feature, receiver feature,
# and edge feature, via a dense layer followed by ReLU.
#
# node_feat: node feature array, length n_nodes * n_node_feat, row-major
# edge_feat: edge feature array, length n_edges * n_edge_feat, row-major
# src: sender node index for each edge, length n_edges (1-based)
# dst: receiver node index for each edge, length n_edges (1-based)
# w_msg: message-layer weight, length n_msg_feat * (2*n_node_feat+n_edge_feat)
# b_msg: message-layer bias, length n_msg_feat
# n_edges: number of edges
# n_node_feat: number of node features
# n_edge_feat: number of edge features
# n_msg_feat: number of message features
# msg_input: scratch array of length 2*n_node_feat + n_edge_feat, reused per edge
# msg_scratch: scratch array of length n_msg_feat, reused per edge
# messages: output array of length n_edges * n_msg_feat, row-major, filled in place
function message_function(node_feat, edge_feat, src, dst, w_msg, b_msg, n_edges, n_node_feat, n_edge_feat, n_msg_feat, msg_input, msg_scratch, messages)
    # size of the concatenated [sender ; receiver ; edge] input vector
    n_in_msg = 2 * n_node_feat + n_edge_feat
    # named zero offset, used wherever a copy starts at the beginning of an array
    zero_off = 0
    # 
    two_n_node_feat = 2 * n_node_feat
    # each edge's message only depends on that edge's own features
    for e = 1:n_edges
        s_node = src[e]
        d_node = dst[e]
        src_off = (s_node - 1) * n_node_feat
        dst_off = (d_node - 1) * n_node_feat
        edge_off = (e - 1) * n_edge_feat
        # assemble [sender feature ; receiver feature ; edge feature]
        copy_segment(node_feat, src_off, msg_input, zero_off, n_node_feat)
        copy_segment(node_feat, dst_off, msg_input, n_node_feat, n_node_feat)
        copy_segment(edge_feat, edge_off, msg_input, two_n_node_feat, n_edge_feat)
        # dense layer plus ReLU produces the message for this edge
        linear_transform(w_msg, b_msg, msg_input, n_in_msg, n_msg_feat, msg_scratch)
        relu_vec(msg_scratch, n_msg_feat, msg_scratch)
        msg_off = (e - 1) * n_msg_feat
        copy_segment(msg_scratch, zero_off, messages, msg_off, n_msg_feat)
    end
    return nothing
end

# aggregate_messages(messages, dst, agg, n_edges, n_msg_feat)
#
# Sums the incoming messages at each receiver node (assumes agg has
# already been cleared with zero_vec by the caller).
#
# messages: message array, length n_edges * n_msg_feat, row-major
# dst: receiver node index for each edge, length n_edges (1-based)
# agg: per-node aggregated message array, length n_nodes * n_msg_feat,
#      row-major, accumulated in place
# n_edges: number of edges
# n_msg_feat: number of message features
function aggregate_messages(messages, dst, agg, n_edges, n_msg_feat)
    # sequential: different edges can share the same receiver node, so
    # this loop carries a dependency through the shared agg array
    for i_seq_e = 1:n_edges
        d_node = dst[i_seq_e]
        msg_off = (i_seq_e - 1) * n_msg_feat
        agg_off = (d_node - 1) * n_msg_feat
        # add this edge's message into its receiver's running total
        for j = 1:n_msg_feat
            agg[agg_off + j] = agg[agg_off + j] + messages[msg_off + j]
        end
    end
    return nothing
end

# update_function(node_feat, agg, w_upd, b_upd, n_nodes, n_node_feat,
#                  n_msg_feat, upd_input, upd_scratch, node_feat_out)
#
# Updates each node's feature vector from its previous feature and its
# aggregated incoming messages, via a dense layer followed by ReLU.
#
# node_feat: node feature array, length n_nodes * n_node_feat, row-major
# agg: per-node aggregated message array, length n_nodes * n_msg_feat, row-major
# w_upd: update-layer weight, length n_node_feat * (n_node_feat + n_msg_feat)
# b_upd: update-layer bias, length n_node_feat
# n_nodes: number of nodes
# n_node_feat: number of node features
# n_msg_feat: number of message features
# upd_input: scratch array of length n_node_feat + n_msg_feat, reused per node
# upd_scratch: scratch array of length n_node_feat, reused per node
# node_feat_out: output array, length n_nodes * n_node_feat, row-major, filled in place
function update_function(node_feat, agg, w_upd, b_upd, n_nodes, n_node_feat, n_msg_feat, upd_input, upd_scratch, node_feat_out)
    # size of the concatenated [old feature ; aggregated message] input
    n_in_upd = n_node_feat + n_msg_feat
    # named zero offset, used wherever a copy starts at the beginning of an array
    zero_off = 0
    # each node's update only depends on that node's own data
    for v = 1:n_nodes
        node_off = (v - 1) * n_node_feat
        agg_off = (v - 1) * n_msg_feat
        # assemble [previous feature ; aggregated incoming message]
        copy_segment(node_feat, node_off, upd_input, zero_off, n_node_feat)
        copy_segment(agg, agg_off, upd_input, n_node_feat, n_msg_feat)
        # dense layer plus ReLU produces the updated node feature
        linear_transform(w_upd, b_upd, upd_input, n_in_upd, n_node_feat, upd_scratch)
        relu_vec(upd_scratch, n_node_feat, upd_scratch)
        copy_segment(upd_scratch, zero_off, node_feat_out, node_off, n_node_feat)
    end
    return nothing
end

# mpnn_layer(node_feat, edge_feat, src, dst, w_msg, b_msg, w_upd, b_upd,
#            n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat,
#            msg_input, msg_scratch, messages, agg, upd_input, upd_scratch,
#            node_feat_out)
#
# Runs one message-passing layer of a Message-Passing Neural Network:
# builds a message per edge, aggregates messages at each receiver node,
# and updates every node feature from its old value and its aggregate.
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
# msg_input: scratch array, length 2*n_node_feat + n_edge_feat
# msg_scratch: scratch array, length n_msg_feat
# messages: scratch array, length n_edges * n_msg_feat
# agg: scratch array, length n_nodes * n_msg_feat
# upd_input: scratch array, length n_node_feat + n_msg_feat
# upd_scratch: scratch array, length n_node_feat
# node_feat_out: output array, length n_nodes * n_node_feat, filled in place
function mpnn_layer(node_feat, edge_feat, src, dst, w_msg, b_msg, w_upd, b_upd, n_nodes, n_edges, n_node_feat, n_edge_feat, n_msg_feat, msg_input, msg_scratch, messages, agg, upd_input, upd_scratch, node_feat_out)
    # total number of scalars in the per-node aggregate array
    n_agg = n_nodes * n_msg_feat
    # start every node's aggregate at zero before summing incoming messages
    zero_vec(agg, n_agg)
    # step 1: compute one message per edge
    message_function(node_feat, edge_feat, src, dst, w_msg, b_msg, n_edges, n_node_feat, n_edge_feat, n_msg_feat, msg_input, msg_scratch, messages)
    # step 2: sum messages at each receiver node
    aggregate_messages(messages, dst, agg, n_edges, n_msg_feat)
    # step 3: update every node feature from its old value and its aggregate
    update_function(node_feat, agg, w_upd, b_upd, n_nodes, n_node_feat, n_msg_feat, upd_input, upd_scratch, node_feat_out)
    return nothing
end