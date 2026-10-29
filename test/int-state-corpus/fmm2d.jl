# fmm2d(s, T, state, heap, pos, off, nx, ny, h, src, rec, out)
#
# First-order fast marching method for the 2D Eikonal equation |grad T| = s,
# with an explicit binary min-heap (decrease-key through a position array).
# The grid is padded with a one-cell ring of cells that are never relaxed, so
# no neighbor lookup needs a bounds check.
#
# s: slowness at every padded grid node, flat array of length nx * ny (independent input)
# T: travel time at every padded grid node, flat array of length nx * ny, filled in place
# state: integer work array of length nx * ny; 0 far, 1 in the heap, 2 accepted
# heap: integer work array of length nx * ny; node indices ordered as a binary min-heap on T
# pos: integer work array of length nx * ny; position of each node in the heap
# off: integer array of length 4 holding the flat neighbor offsets -1, 1, -nx, nx
# nx: padded grid size along the fast index
# ny: padded grid size along the slow index
# h: grid spacing
# src: flat index of the source node
# rec: flat index of the receiver node
# out: length-1 output array; out[1] receives the travel time at the receiver
function fmm2d(s, T, state, heap, pos, off, nx, ny, h, src, rec, out)
    n = nx * ny
    big = 1.0e30
    # every node starts far from the source, outside the heap
    for u = 1:n
        T[u] = big
        state[u] = 0
        pos[u] = 0
    end
    # padding ring: marked accepted with an infinite time, so it is never relaxed
    # and never lowers an upwind minimum
    for i = 1:nx
        state[i] = 2
        state[(ny - 1) * nx + i] = 2
    end
    for j = 1:ny
        state[(j - 1) * nx + 1] = 2
        state[j * nx] = 2
    end
    # the heap starts with the source only
    T[src] = 0.0
    state[src] = 1
    heap[1] = src
    pos[src] = 1
    nheap = 1
    # the order in which nodes are accepted depends on the slowness model,
    # so the number of iterations is only known once the loop has run
    while nheap > 0
        # pop the root: the narrow-band node with the smallest travel time
        u = heap[1]
        state[u] = 2
        last = heap[nheap]
        nheap -= 1
        if nheap > 0
            # move the last heap entry to the root and sift it down
            heap[1] = last
            pos[last] = 1
            tl = T[last]
            k = 1
            while 2 * k <= nheap
                c = 2 * k
                # pick the smaller of the two children
                if c < nheap
                    c1 = heap[c]
                    c2 = heap[c + 1]
                    if T[c2] < T[c1]
                        c = c + 1
                    end
                end
                cn = heap[c]
                if T[cn] < tl
                    heap[k] = cn
                    pos[cn] = k
                    heap[c] = last
                    pos[last] = c
                    k = c
                else
                    # heap order restored: forcing k past the last parent ends the loop
                    k = nheap
                end
            end
        end
        # relax the four neighbors of the accepted node
        for d = 1:4
            v = u + off[d]
            if state[v] < 2
                # smallest accepted neighbor time along each axis
                ta = big
                tb = big
                for e = 1:2
                    w = v + off[e]
                    if state[w] == 2
                        ta = min(ta, T[w])
                    end
                end
                for e = 3:4
                    w = v + off[e]
                    if state[w] == 2
                        tb = min(tb, T[w])
                    end
                end
                lo = min(ta, tb)
                hi = max(ta, tb)
                f = h * s[v]
                # one-sided update, replaced by the two-sided quadratic solution
                # when both axes carry information
                cand = lo + f
                if hi - lo < f
                    cand = 0.5 * (lo + hi + sqrt(2.0 * f * f - (hi - lo) * (hi - lo)))
                end
                if cand < T[v]
                    T[v] = cand
                    # a far node enters the heap at the end
                    if state[v] == 0
                        nheap += 1
                        heap[nheap] = v
                        pos[v] = nheap
                        state[v] = 1
                    end
                    # sift the lowered key up toward the root
                    k = pos[v]
                    while k > 1
                        par = div(k, 2)
                        pn = heap[par]
                        if T[pn] > cand
                            heap[k] = pn
                            pos[pn] = k
                            heap[par] = v
                            pos[v] = par
                            k = par
                        else
                            # heap order restored: k = 1 ends the loop
                            k = 1
                        end
                    end
                end
            end
        end
    end
    out[1] = T[rec]
    return nothing
end
