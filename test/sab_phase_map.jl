# Builds sabotage_phases.tsv from the SABMAP lines of a sabotage run (v0.4.74, tests only).
#
#   julia sab_phase_map.jl OUT.tsv LOG...
#
# Produce the logs by running the sabotage phase of the multi-phase test files with SAB_MAP=1 SAB_NOFF=1: every check phase of each case runs
# alone and prints  SABMAP<TAB>file<TAB>label<TAB>phase<TAB>failures<TAB>seconds.  A log may hold those lines as they are, or joined by '@'
# on one line that starts with SABMAPS (the form a rome chain task leaves in its log).
# For each (file, label) the phase that has failures and ran fastest is written. A case no phase catches gets no entry, so it runs every phase
# and shows up as NOT CAUGHT.
function sab_phase_map_main(out::String, logs::Vector{String})
    best = Dict{Tuple{String,String},Tuple{Int,Float64}}()
    seen = Set{Tuple{String,String}}()
    for lg in logs
        for l in eachline(lg)
            startswith(l, "SABMAPS") && (l = strip(l[length("SABMAPS")+1:end]))
            for rec in split(l, '@')
                p = split(rec, '\t')
                (length(p) == 6 && p[1] == "SABMAP") || continue
                key = (String(p[2]), String(p[3]))
                push!(seen, key)
                i = parse(Int, p[4]); n = parse(Int, p[5]); t = parse(Float64, p[6])
                n > 0 || continue
                (haskey(best, key) && best[key][2] <= t) || (best[key] = (i, t))
            end
        end
    end
    open(out, "w") do io
        println(io, "# file<TAB>label<TAB>phase that catches the case (written by sab_phase_map.jl)")
        for key in sort(collect(keys(best)))
            println(io, key[1], '\t', key[2], '\t', best[key][1])
        end
    end
    println(length(seen), " cases seen, ", length(best), " caught by some phase, ", length(seen) - length(best), " not caught")
    return length(seen) - length(best)
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(sab_phase_map_main(ARGS[1], String.(ARGS[2:end])) == 0 ? 0 : 1)
end
