# Writes the family F3 subjects (deep nests, depth 3 to 6) into test/battle-corpus.
# The files are committed.  Run again only to change the pattern list.
include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
include(joinpath(@__DIR__, "battle_lib.jl"))

const BT_NEST_SPECS = vcat(
    [(p, t, f) for p in ["III", "IIS", "ISI", "ISS", "SII", "SIS", "SSI", "SSS"] for (t, f) in ((false, false), (false, true), (true, false))],
    [(p, false, f) for p in ["IIII", "ISIS", "SIIS", "IISI", "SSII", "ISSI", "SISI", "IIIS"] for f in (false, true)],
    [(p, false, f) for p in ["IIIII", "ISISI", "SIIIS", "IISII", "SSSSI", "ISSSS"] for f in (false, true)],
    [(p, false, f) for p in ["IIIIII", "ISISIS", "SIIIIS", "IIISII", "SSSSSI", "ISSSSS"] for f in (false, true)],
)

function bt_make_nests(dir = joinpath(@__DIR__, "battle-corpus"))
    n = 0
    for (p, t, f) in BT_NEST_SPECS
        lead = something(findfirst(!=('I'), collect(p)), length(p) + 1) - 1
        (f && (lead < 2 || t)) && continue
        name = "f3_nest_" * p * (t ? "_tri" : "") * (f ? "_fz" : "")
        src, classes = bt_nest_source(name, p; tri = t, fuse = f)
        fn = bt_parse_fn(src)
        exp = bt_expected(fn, classes)
        tab = ["# LOOP $(v) $(classes[v]) $(exp[v])" for v in sort(collect(keys(classes)))]
        desc = "Nest $p" * (t ? ", triangular bounds" : "") * (f ? ", leading independent loops pre-fused" : "") * "."
        open(joinpath(dir, name * ".jl"), "w") do io
            println(io, "# ", name); println(io, "# ", desc); foreach(l -> println(io, l), tab); print(io, src)
        end
        n += 1
    end
    return n
end
if abspath(PROGRAM_FILE) == @__FILE__
    println("wrote ", bt_make_nests(), " subjects")
end
