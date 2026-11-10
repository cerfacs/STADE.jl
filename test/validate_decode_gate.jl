# Regression for the decoded-loop-variable proof of the offload gate (defect B4 of the offload battle test).
# skill-stade rule 2 fuses a nest of independent loops into `for fz = 1:n1 * n2` and decodes the indices with div and mod. When the body also
# holds a sequential inner loop (a read at `l3 - 1`), two iterations of fz touch different elements only if the decoded indices identify fz:
# every access holds ALL the decoded indices at fixed dimensions, the radices match the decode, and the range is exactly their product.
# The gate accepts that and nothing else. Each wrong variant below must stay on the host.
# Reads the gate only. No device is used.
include(joinpath(@__DIR__, "validate_battle_status.jl"))

const DG_BODY = """
        for l3 = 2:n3
            y[l1, l2, l3] = 0.5 * y[l1, l2, l3 - 1] + x[l1, l2, l3]
        end
"""
dg_kernel(loophdr, decode, body = DG_BODY; pre = "") = "function g(x, y, a, n1, n2, n3)\n$(pre)    for $(loophdr)\n$(decode)$(body)    end\n    return nothing\nend"

const DG_DEC2m = "        l1 = mod(div(fz - 1, n2), m) + 1\n        l2 = mod(fz - 1, n2) + 1\n"
const DG_DEC2 = "        l1 = mod(div(fz - 1, n2), n1) + 1\n        l2 = mod(fz - 1, n2) + 1\n"
# (name, source, accepted)
const DG_CASES = [
    ("two digits", dg_kernel("fz = 1:n1 * n2", DG_DEC2), true),
    ("temporary for fz - 1", dg_kernel("fz = 1:n1 * n2", "        t0 = fz - 1\n        l1 = mod(div(t0, n2), n1) + 1\n        l2 = mod(t0, n2) + 1\n"), true),
    ("reversed range", dg_kernel("fz = n1 * n2:-1:1", DG_DEC2), true),
    ("three digits", """
function g(x, y, a, n1, n2, n3)
    for fz = 1:n1 * n2 * n3
        l1 = mod(div(fz - 1, n2 * n3), n1) + 1
        l2 = mod(div(fz - 1, n3), n2) + 1
        l3 = mod(fz - 1, n3) + 1
        for l4 = 2:4
            y[l1, l2, l3, l4] = 0.5 * y[l1, l2, l3, l4 - 1] + x[l1, l2, l3, l4]
        end
    end
    return nothing
end""", true),
    ("range too long", dg_kernel("fz = 1:2 * n1 * n2", DG_DEC2), false),
    ("range is one radix", dg_kernel("fz = 1:n1", DG_DEC2), false),
    ("range starts at 2", dg_kernel("fz = 2:n1 * n2", DG_DEC2), false),
    ("wrong divisor", dg_kernel("fz = 1:n1 * n2", "        l1 = mod(div(fz - 1, n3), n1) + 1\n        l2 = mod(fz - 1, n2) + 1\n"), false),
    ("wrong radix of the last digit", dg_kernel("fz = 1:n1 * n2", "        l1 = mod(div(fz - 1, n2), n1) + 1\n        l2 = mod(fz - 1, n3) + 1\n"), false),
    ("one digit not indexed", dg_kernel("fz = 1:n1 * n2", DG_DEC2, "        for l3 = 2:n3\n            y[l1, l3] = 0.5 * y[l1, l3 - 1] + x[l1, l3]\n        end\n"), false),
    ("read at another digit", dg_kernel("fz = 1:n1 * n2", DG_DEC2, "        for l3 = 2:n3\n            y[l1, l2, l3] = 0.5 * y[l1, l2 - 1, l3] + x[l1, l2, l3]\n        end\n"), false),
    ("digit set twice", dg_kernel("fz = 1:n1 * n2", DG_DEC2, DG_BODY * "        l1 = 1\n"), false),
    ("digit rewritten after its use", dg_kernel("fz = 1:n1 * n2", "        l2 = mod(fz - 1, n2) + 1\n        l1 = 1\n", DG_BODY * "        l1 = mod(div(fz - 1, n2), n1) + 1\n"), false),
    ("radix is a local scalar", dg_kernel("fz = 1:m * n2", DG_DEC2m; pre = "    m = n1 + 0\n"), false),
    ("digit used before it is set", dg_kernel("fz = 1:n1 * n2", "", DG_BODY * DG_DEC2; pre = "    l1 = 1\n    l2 = 1\n"), false),
    ("radix changes in the loop", dg_kernel("fz = 1:n1 * n2", DG_DEC2, DG_BODY * "        n2 = n2 + 0\n"), false),
]

function validate_decode_gate(; verbose = true)
    bad = 0; checks = 0
    for (name, src, accept) in DG_CASES
        checks += 1
        f = try Meta.parse(src) catch e; (bad += 1; verbose && println("NOT OK ", name, ": does not parse")); continue end
        k = try STADE.parse_kernel(f) catch e; (bad += 1; verbose && println("NOT OK ", name, ": parse_kernel: ", first(sprint(showerror, e), 100))); continue end
        st = first(s for s in k.body if s.kind == :for)
        g = STADE.cgen_split_loop_gate(st.body, st.var; bounds = (st.lo, st.step, st.hi), protect = Set{Symbol}(k.sig.args))
        ok = (g.reason === :ok)
        if ok != accept
            bad += 1
            verbose && println("NOT OK ", name, ": gate says ", g.reason, " (array ", g.array, "), expected ", accept ? "ok" : "a refusal")
        end
    end
    println(bad == 0 ? "all $checks decode gate checks passed" : "$bad of $checks decode gate checks NOT OK")
    return bad == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    validate_decode_gate() || exit(1)
end
