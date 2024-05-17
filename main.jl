using DelimitedFiles, ExaModelsPower, CUSOLVERRF, MadNLP, MadNLPGPU, CUDA, NLPModelsIpopt, HSL_jll, JLD2

tol = 1e-4

DATA = [
    (
        "case30",
        "data/case30/pglib_opf_case30_ieee.m",
        "data/case30/case30.bus",
        "data/case30/halfhour_30.Pd",
        "data/case30/halfhour_30.Qd"
    ),
    (
        "case30",
        "data/case30/pglib_opf_case30_ieee.m",
        "data/case30/case30.bus",
        "data/case30/halfhour_30.Pd",
        "data/case30/halfhour_30.Qd"
    ),
   (
       "case118",
       "data/case118/pglib_opf_case118_ieee.m",
       "data/case118/case118.bus",
       "data/case118/case118_onehour_168.Pd",
       "data/case118/case118_onehour_168.Qd"
   ),
   ( # out of memoery
       "case118",
       "data/case118/pglib_opf_case118_ieee.m",
       "data/case118/case118.bus",
       "data/case118/case118_oneminute_10080.Pd",
       "data/case118/case118_oneminute_10080.Qd"
   ),
    (
        "case1354",
        "data/case1354/pglib_opf_case1354_pegase.m",
        "data/case1354/case1354pegase.bus",
        "data/case1354/halfhour_30.Pd",
        "data/case1354/halfhour_30.Qd"
    ),
    (
        "case9241",
        "data/case9241/pglib_opf_case9241_pegase.m",
        "data/case9241/case9241pegase.bus",
        "data/case9241/halfhour_30.Pd",
        "data/case9241/halfhour_30.Qd"
    ),
    (
        "case9241",
        "data/case9241/pglib_opf_case9241_pegase.m",
        "data/case9241/case9241pegase.bus",
        "data/case9241/halfhour_180.Pd",
        "data/case9241/halfhour_180.Qd"
    ),
]
results_file = "results/results_$(gethostname()).jld2"

results = isfile(results_file) ? load(results_file) : Dict{String,Any}()
if isinteractive()
      (caseid, cpu, gpu) = (2,false, true)
elseif length(ARGS) == 3
    caseid = parse(Int, ARGS[1])
    cpu = parse(Bool, ARGS[2])
    gpu = parse(Bool, ARGS[3])
else
    error("Invalid number of arguments")
end
println("caseid=$caseid, cpu=$cpu, gpu=$gpu")

function run_case(caseid::Int, cpu, gpu)
    (name, net, bus, Pd, Qd) = DATA[caseid]
    periods =  size(readdlm(Qd),2)
    println("Running case ($name, $net, $bus, $Pd, $Qd)")
    (tcpu, tgpu, cpuiter, gpuiter, statuscpu, statusgpu) = (
        0.0, 0.0, 0, 0, 0, 0
    )
    if cpu
        model, vars = mpopf_model(net, bus, Pd, Qd)
        tcpu = @elapsed rcpu = ipopt(
            model;
            linear_solver="ma27",
            tol = tol,
            bound_relax_factor = tol,
            dual_inf_tol = 10000.0,
            constr_viol_tol = 10000.0,
            compl_inf_tol = 10000.0,
            honor_original_bounds = "no",
            max_wall_time = 600.
            # output_file = "jump_output",
            # print_timing_statistics = "yes"
        )
        cpuiter = rcpu.iter
        statuscpu = rcpu.status
    end

    if gpu 
        model, vars = mpopf_model(net, bus, Pd, Qd; backend = CUDABackend())
        tgpu = @elapsed rgpu = madnlp(
            model;
            tol = tol,
            max_wall_time = 600.
        )
        gpuiter = rgpu.iter
        statusgpu = rgpu.status
    end

    if caseid != 1 # first case is for JIT compilation
        results["$caseid"] = (
                name = name,
                nvar = model.meta.nvar,
                ncon = model.meta.ncon,
                tcpu = tcpu,
                tgpu = tgpu,
                statuscpu = statuscpu == :first_order,
                statusgpu = statusgpu == MadNLP.SOLVE_SUCCEEDED,
                itercpu = cpuiter,
                itergpu = gpuiter
            )
        save(results_file, results)
    end
end

# precompilation
run_case(1, cpu, gpu)
run_case(caseid, cpu, gpu)

