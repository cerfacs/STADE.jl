# Minimal illustration of why a running counter isn't sufficient for GPU
# porting, even though it's simpler to generate than a loop-variable-
# derived index.
#
# Setup: overwrite v[i] for i in 1:n, saving the old value first (this
# stands in for one push!/pop! site inside one loop -- e.g. du_stack in
# advection_b). Later, restore v[i] from the saved value.
#
# On a real GPU launch, neither the "save" pass nor the "restore" pass is
# guaranteed to visit i in any particular order -- each thread runs
# independently and the scheduler is free to interleave them however it
# likes. Nothing in the semantics of a parallel launch requires the
# restore pass to revisit indices in the same relative order the save
# pass used. We simulate that here by literally giving the two passes
# different visitation orders -- a legitimate thing for a GPU scheduler
# to do, not a contrived worst case.

n = 6
v_orig = collect(1.0:n)

forward_order  = collect(1:n)          # order the "save" pass visits i in
backward_order = [1, 3, 2, 6, 4, 5]    # a DIFFERENT order for the "restore" pass

# ---- counter-based scheme: push!/pop! replaced by an incrementing index ----
v = deepcopy(v_orig)
buf_counter = Vector{Float64}(undef, n)
idx = 0
for i in forward_order
    global idx += 1
    buf_counter[idx] = v[i]      # "push!(stack, v[i])"
    v[i] = 2 * v[i]              # the overwrite being protected
end

restored_counter = deepcopy(v)
idx = n + 1
for i in backward_order
    global idx -= 1
    restored_counter[i] = buf_counter[idx]   # "v[i] = pop!(stack)"
end

# ---- closed-form scheme: index = i itself, no counter at all ----
v2 = deepcopy(v_orig)
buf_indexed = Vector{Float64}(undef, n)
for i in forward_order
    buf_indexed[i] = v2[i]       # store directly at position i
    v2[i] = 2 * v2[i]
end

restored_indexed = deepcopy(v2)
for i in backward_order
    restored_indexed[i] = buf_indexed[i]     # load directly from position i
end

println("original values:        ", v_orig)
println("counter-based restore:  ", restored_counter, "   correct = ", restored_counter == v_orig)
println("index-based restore:    ", restored_indexed, "   correct = ", restored_indexed == v_orig)