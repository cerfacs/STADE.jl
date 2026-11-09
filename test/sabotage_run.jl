# Shared by the sabotage phase of the validate_*.jl scripts (v0.4.74, tests only; nothing here is part of STADE.jl).
#
# A sabotage case rewrites one guard in a copy of src/STADE.jl and expects some check to fail. Two helpers make that cheaper
# without changing what is expected:
#
#   sab_read(script)        runs the sabotaged script in a fresh Julia process and returns its output. The first line that starts with
#                           "FAIL" ends the run ("RESULT 1" is returned): the case is caught and the checks after it add nothing.
#                           A run that never prints FAIL is read to the end, so a case that is NOT caught still runs every check.
#                           SAB_NOFF=1 reads to the end in every case.
#   sab_phase_sum(file, label, fs)
#                           runs the check phases `fs` of one test file. When sabotage_phases.tsv names the phase that catches
#                           (file, label), only that phase runs. A case without an entry runs every phase, and SAB_ALL_PHASES=1
#                           runs every phase for every case.
#
# sabotage_phases.tsv is written by sab_phase_map.jl from a run with SAB_MAP=1 (each phase is run alone and its failure count and time are
# printed as SABMAP lines). To refresh it after a case or a check phase changes, run that tool; a stale entry can only make a case run one phase
# that no longer catches it, and the case then shows up as NOT CAUGHT.
isdefined(@__MODULE__, :sab_read) || begin

function sab_read(script::String)
    noff = get(ENV, "SAB_NOFF", "") == "1"
    buf = IOBuffer()
    try
        proc = open(Base.pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull))
        for l in eachline(proc)
            startswith(l, "SABMAP") && println(l)
            if !noff && startswith(l, "FAIL")
                kill(proc)
                return "RESULT 1\n"
            end
            println(buf, l)
        end
        wait(proc)
        success(proc) || return "ERR"
    catch
        return "ERR"
    end
    return String(take!(buf))
end

function sab_phase_map()
    m = Dict{Tuple{String,String},Int}()
    path = joinpath(@__DIR__, "sabotage_phases.tsv")
    isfile(path) || return m
    for l in eachline(path)
        (isempty(l) || startswith(l, "#")) && continue
        p = split(l, '\t')
        length(p) == 3 && (m[(String(p[1]), String(p[2]))] = parse(Int, p[3]))
    end
    return m
end

function sab_phase_sum(file::String, label::String, fs)
    if get(ENV, "SAB_MAP", "") == "1"
        tot = 0
        for (i, f) in enumerate(fs)
            t0 = time()
            n = try f() catch; 1 end
            println("SABMAP\t", file, "\t", label, "\t", i, "\t", n, "\t", round(time() - t0, digits = 1))
            tot += n
        end
        return tot
    end
    idx = get(ENV, "SAB_ALL_PHASES", "") == "1" ? 0 : get(sab_phase_map(), (file, label), 0)
    idx in 1:length(fs) && return fs[idx]()
    return sum(f() for f in fs)
end

end
