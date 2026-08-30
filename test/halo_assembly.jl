# halo_assembly(out, u, w, b, buf, n_cell, n_a, n_b, n_out)
#
# Cumulative-assembly shape with a loop-carried read. This is mpnn's buffer
# pattern -- each cell fills its own slice of buf in two pieces, then a dense
# layer consumes the whole slice as one contiguous run -- with one deliberate
# change: the dense layer reads the PREVIOUS cell's slice, not its own.
#
# That single offset is the whole point of the kernel. Cell c reads what cell
# c-1 wrote on the previous iteration, so the outer loop carries a genuine
# read-after-write and MUST NOT be split across threads. It is the negative
# control for cgen_array_private_to_loop's pattern 3 (cumulative assembly):
# every structural cue that makes mpnn's loop offloadable is present here, and
# only the base of the reading group differs, so an analysis that matched on
# shape rather than on the actual index would accept it and race.
#
# It also exercises cgen_snapshot_save_dead on the accepting side. buf is
# overwritten after being read, so the adjoint emits a snapshot save for it and
# the HVP separates that save from its overwrite by the interleaved shadow twin
# -- the branch that was missing until mpnn's HVP was found running one kernel
# per graph edge. Here the elision is correct and the loop must still stay on
# the host, which is the combination worth pinning: the relaxation must not be
# what lets an unsafe loop through.
#
# buf carries a leading zero slice so cell 1 reads slice 0 rather than
# out-of-bounds, which keeps every index affine and avoids a max() the
# injectivity check would have to reason about.
#
# out: output array of length n_cell * n_out, filled in place
# u: input array of length n_cell * n_a
# w: dense weights of length n_out * (n_a + n_b)
# b: dense bias of length n_out
# buf: scratch of length (n_cell + 1) * (n_a + n_b), caller-provided
# n_cell: number of cells
# n_a: width of the first assembled piece
# n_b: width of the second assembled piece
# n_out: dense output width
function halo_assembly(out, u, w, b, buf, n_cell, n_a, n_b, n_out)
    n_w = n_a + n_b
    for c = 1:n_cell
        off = c * n_w
        prev = (c - 1) * n_w
        # piece one of this cell's slice
        for k = 1:n_a
            buf[off + k] = u[(c - 1) * n_a + k]
        end
        # piece two, contiguous with piece one -- together they cover n_w
        for k = 1:n_b
            buf[off + n_a + k] = u[(c - 1) * n_a + 1] * 0.5
        end
        # dense layer over the PREVIOUS cell's slice: this is the dependence
        for o = 1:n_out
            s = b[o]
            for i_i = 1:n_w
                s = s + w[(o - 1) * n_w + i_i] * buf[prev + i_i]
            end
            out[(c - 1) * n_out + o] = s
        end
    end
    return nothing
end
