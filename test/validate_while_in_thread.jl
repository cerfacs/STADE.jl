# Regression for a sequential `while` inside a device thread (defect B7 of the offload battle test, v0.5.8).
# An independent loop whose body holds a `while` is offloaded when every scalar it touches is private to the iteration and every array
# access is owned by the iteration. The `while` runs in the thread, like an inner `for` with an unknown trip count.
# The cases below that must stay on the host each isolate ONE guard of the proof; the sabotage runs (see the report) remove the guard and
# the case flips to a kernel:
#   M  a local scalar read by the condition before the iteration defines it   (cgen_use_before_def, condition read)
#   N  an array the condition reads, written at another index                 (cgen_collect_array_accesses!, condition refs; cgen_array_private_to_loop)
#   O  the same through a scalar index                                        (same)
#   R  the same inside an inner `for`                                         (cgen_deep_array_occurrences!, condition read)
# The free variables of the kernel must include the symbols of the condition: the first device run of v0.5.8 failed to compile because
# `tol`, read only by the condition, was not passed to the kernel.
# Reads generated code. No device is used. The device check is a separate job.
include(joinpath(@__DIR__, "validate_battle_status.jl"))

const WT_CASES = [
    # (name, source, offloaded)
    ("safe: private scalar, own element", "function g(x, y, tol, n)\n for i1 = 1:n\n r1 = x[i1]\n while r1 > tol\n r1 = 0.5 * r1\n end\n y[i1] = r1\n end\n return nothing\nend", true),
    ("safe: atomic accumulate through a gathered index", "function g(x, y, idx, tol, n)\n for i1 = 1:n\n j1 = idx[i1]\n r1 = x[i1]\n while r1 > tol\n y[j1] = y[j1] + 1.0\n r1 = 0.5 * r1\n end\n end\n return nothing\nend", true),
    ("scalar set only inside the while, read after", "function g(x, y, tol, n)\n q1 = 0.0\n for i1 = 1:n\n r1 = x[i1]\n while r1 > tol\n r1 = 0.5 * r1\n q1 = r1\n end\n y[i1] = q1\n end\n return nothing\nend", false),
    ("M condition reads an undefined local", "function g(x, y, tol, n)\n q1 = 1.0\n for i1 = 1:n\n r1 = x[i1]\n while q1 > tol\n r1 = 0.5 * r1\n q1 = r1\n end\n q1 = x[i1]\n y[i1] = r1\n end\n return nothing\nend", false),
    ("carried condition scalar", "function g(x, y, tol, n)\n c1 = 1.0\n for i1 = 1:n\n while c1 > tol\n c1 = 0.5 * c1\n end\n y[i1] = c1 + x[i1]\n end\n return nothing\nend", false),
    ("outer counter bumped in the while", "function g(x, y, tol, n)\n tot = 0.0\n for i1 = 1:n\n r1 = x[i1]\n while r1 > tol\n r1 = 0.5 * r1\n tot = tot + 1.0\n end\n y[i1] = r1\n end\n y[1] = tot\n return nothing\nend", false),
    ("N condition reads the neighbour element", "function g(x, y, tol, n)\n for i1 = 1:n-1\n r1 = x[i1]\n while r1 > y[i1 + 1]\n r1 = 0.5 * r1\n end\n y[i1] = r1\n end\n return nothing\nend", false),
    ("O the same through a scalar index", "function g(x, y, tol, n)\n for i1 = 1:n-1\n j1 = i1 + 1\n r1 = x[i1]\n while r1 > y[j1]\n r1 = 0.5 * r1\n end\n y[i1] = r1\n end\n return nothing\nend", false),
    ("R the same inside an inner for", "function g(x, y, tol, n, m)\n for i1 = 1:n-1\n r1 = x[i1]\n for c1 = 1:m\n while r1 > y[i1 + 1]\n r1 = 0.5 * r1\n end\n end\n y[i1] = r1\n end\n return nothing\nend", false),
    ("previous element read after the while", "function g(x, y, tol, n)\n for i1 = 2:n\n r1 = x[i1]\n while r1 > tol\n r1 = 0.5 * r1\n end\n y[i1] = r1 + y[i1 - 1]\n end\n return nothing\nend", false),
    ("plain write to a fixed index in the while", "function g(x, y, tol, n)\n for i1 = 1:n\n r1 = x[i1]\n while r1 > tol\n y[1] = r1\n r1 = 0.5 * r1\n end\n end\n return nothing\nend", false),
    ("gathered accumulate and gathered plain write", "function g(x, y, idx, idx2, tol, n)\n for i1 = 1:n\n j1 = idx[i1]\n k1 = idx2[i1]\n r1 = x[i1]\n while r1 > tol\n y[j1] = y[j1] + 1.0\n y[k1] = 0.0\n r1 = 0.5 * r1\n end\n end\n return nothing\nend", false),
]

function validate_while_in_thread(; verbose = true)
    bad = 0; checks = 0
    for (name, src, accept) in WT_CASES, be in (:cuda, :jacc)
        checks += 1
        f = Meta.parse(src)
        pl = try bt_plan(f, be, :pinned) catch e; (bad += 1; verbose && println("NOT OK ", name, " ", be, ": plan error ", first(sprint(showerror, e), 100))); continue end
        got = length(pl.kernels) > 0
        if got != accept
            bad += 1
            verbose && println("NOT OK ", name, " ", be, ": ", got ? "offloaded" : "host", ", expected ", accept ? "offload" : "host")
        end
    end
    # the condition's symbols are free variables of the kernel
    checks += 1
    k = STADE.parse_kernel(Meta.parse(WT_CASES[1][2]))
    st = first(s for s in k.body if s.kind == :for)
    :tol in STADE.cgen_free_vars(st, st.var) || (bad += 1; verbose && println("NOT OK free variables of the loop miss the symbol of the while condition"))
    # a kernel with a `while` in a thread passes `tol` on the launch
    pl = bt_plan(Meta.parse(WT_CASES[1][2]), :cuda, :pinned)
    checks += 1
    occursin("tol", string(pl.kernels[1].args[1])) || (bad += 1; verbose && println("NOT OK the device kernel does not take `tol`"))
    println(bad == 0 ? "all $checks while-in-thread checks passed" : "$bad of $checks while-in-thread checks NOT OK")
    return bad == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    validate_while_in_thread() || exit(1)
end
