CUDA.@time @testset "CUDA Extension" begin
    # re-define the bath (make the matrix smaller)
    λ = 0.01
    W = 0.5
    kT = 0.5
    μ = 0
    N = 3
    tier = 3

    # System Hamiltonian
    Hsys = Qobj(zeros(ComplexF64, 2, 2))

    # system-bath coupling operator
    Qb = sigmax()
    Qf = sigmam()

    E = Qobj(rand(ComplexF64, 2, 2))
    e_ops = [E]

    # initial state
    ψ0 = basis(2, 1)

    Bbath = Boson_DrudeLorentz_Pade(Qb, λ, W, kT, N)
    Fbath = Fermion_Lorentz_Pade(Qf, λ, μ, W, kT, N)

    # Solving time Evolution
    ## Schrodinger HEOMLS
    L_cpu = M_S(Hsys; verbose = false)
    L_gpu = cu(L_cpu)
    sol_cpu = heomsolve(L_cpu, ψ0, [0, 10]; e_ops = e_ops, verbose = false)
    sol_gpu = heomsolve(L_gpu, ψ0, [0, 10]; e_ops = e_ops, verbose = false)
    @test L_gpu.data.A isa CUDA.CUSPARSE.CuSparseMatrixCSC{ComplexF64,Int32}
    @test all(isapprox.(sol_cpu.expect[1, :], sol_gpu.expect[1, :], atol = 1e-4))
    @test isapprox(getRho(sol_cpu.ados[end]), getRho(sol_gpu.ados[end]), atol = 1e-4)

    ## Boson HEOMLS
    L_cpu = M_Boson(Hsys, tier, Bbath; verbose = false)
    L_gpu = cu(L_cpu)
    sol_cpu = heomsolve(L_cpu, ψ0, [0, 10]; e_ops = e_ops, verbose = false)
    sol_gpu = heomsolve(L_gpu, ψ0, [0, 10]; e_ops = e_ops, verbose = false)
    @test L_gpu.data.A isa CUDA.CUSPARSE.CuSparseMatrixCSC{ComplexF64,Int32}
    @test all(isapprox.(sol_cpu.expect[1, :], sol_gpu.expect[1, :], atol = 1e-4))
    @test isapprox(getRho(sol_cpu.ados[end]), getRho(sol_gpu.ados[end]), atol = 1e-4)

    ## Fermion HEOMLS
    L_cpu = M_Fermion(Hsys, tier, Fbath; verbose = false)
    L_gpu = cu(L_cpu)
    sol_cpu = heomsolve(L_cpu, ψ0, [0, 10]; e_ops = e_ops, verbose = false)
    sol_gpu = heomsolve(L_gpu, ψ0, [0, 10]; e_ops = e_ops, verbose = false)
    @test L_gpu.data.A isa CUDA.CUSPARSE.CuSparseMatrixCSC{ComplexF64,Int32}
    @test all(isapprox.(sol_cpu.expect[1, :], sol_gpu.expect[1, :], atol = 1e-4))
    @test isapprox(getRho(sol_cpu.ados[end]), getRho(sol_gpu.ados[end]), atol = 1e-4)

    ## Boson Fermion HEOMLS
    L_cpu = M_Boson_Fermion(Hsys, tier, tier, Bbath, Fbath; verbose = false)
    L_gpu = cu(L_cpu)
    tlist = 0:1:10
    sol_cpu = heomsolve(L_cpu, ψ0, tlist; e_ops = e_ops, saveat = tlist, verbose = false)
    sol_gpu = heomsolve(L_gpu, ψ0, tlist; e_ops = e_ops, saveat = tlist, verbose = false)
    @test L_gpu.data.A isa CUDA.CUSPARSE.CuSparseMatrixCSC{ComplexF64,Int32}
    @test all(isapprox.(sol_cpu.expect[1, :], sol_gpu.expect[1, :], atol = 1e-4))
    for i in 1:length(tlist)
        @test isapprox(getRho(sol_cpu.ados[i]), getRho(sol_gpu.ados[i]), atol = 1e-4)
    end

    # SIAM
    ϵ = -5
    U = 10
    σm = sigmam() ## σ-
    σz = sigmaz() ## σz
    II = qeye(2)  ## identity matrix
    d_up = tensor(σm, II)
    d_dn = tensor(-1 * σz, σm)
    ψ0 = tensor(basis(2, 0), basis(2, 0))
    Hsys = ϵ * (d_up' * d_up + d_dn' * d_dn) + U * (d_up' * d_up * d_dn' * d_dn)
    Γ = 2
    μ = 0
    W = 10
    kT = 0.5
    N = 5
    tier = 3
    bath_up = Fermion_Lorentz_Pade(d_up, Γ, μ, W, kT, N)
    bath_dn = Fermion_Lorentz_Pade(d_dn, Γ, μ, W, kT, N)
    bath_list = [bath_up, bath_dn]

    ## solve stationary state
    L_even_cpu = M_Fermion(Hsys, tier, bath_list; verbose = false)
    L_even_gpu = cu(L_even_cpu)
    ados_cpu = steadystate(L_even_cpu; verbose = false)
    ados_gpu1 = steadystate(L_even_gpu; verbose = false)
    ados_gpu2 = steadystate(CUDA.CUSPARSE.CuSparseMatrixCSR(L_even_cpu); verbose = false)
    ados_gpu3 = steadystate(L_even_gpu, ψ0, 10; verbose = false)
    @test L_even_gpu.data.A isa CUDA.CUSPARSE.CuSparseMatrixCSC{ComplexF64,Int32}
    @test all(isapprox.(ados_cpu.data, ados_gpu1.data; atol = 1e-6))
    @test all(isapprox.(ados_cpu.data, ados_gpu2.data; atol = 1e-6))
    @test all(isapprox.(ados_cpu.data, ados_gpu3.data; atol = 1e-6))

    ## solve density of states
    ωlist = -5:0.5:5
    L_odd_cpu = M_Fermion(Hsys, tier, bath_list, ODD; verbose = false)
    L_odd_gpu_32 = cu(L_odd_cpu, word_size = Val(32))
    L_odd_gpu_64 = cu(L_odd_cpu, word_size = Val(64))
    dos_cpu = DensityOfStates(L_odd_cpu, ados_cpu, d_up, ωlist; verbose = false)
    dos_gpu_32 = DensityOfStates(
        L_odd_gpu_32,
        ados_cpu,
        d_up,
        ωlist;
        verbose = false,
        solver = KrylovJL_BICGSTAB(rtol = 1.0f-12, atol = 1.0f-14), # somehow KrylovJL_GMRES doesn't work for Float32 (it takes forever to solve)
    )
    dos_gpu_64 = DensityOfStates(L_odd_gpu_64, ados_cpu, d_up, ωlist; verbose = false)
    @test L_odd_gpu_32.data.A isa CUDA.CUSPARSE.CuSparseMatrixCSC{ComplexF32,Int32}
    @test L_odd_gpu_64.data.A isa CUDA.CUSPARSE.CuSparseMatrixCSC{ComplexF64,Int32}
    for (i, ω) in enumerate(ωlist)
        @test dos_cpu[i] ≈ dos_gpu_32[i] atol = 1e-6
        @test dos_cpu[i] ≈ dos_gpu_64[i] atol = 1e-6
    end

    ## I/O-HEOM functionality
    tier = 12
    Δ = 2 * π
    Γ = 0.1 * Δ
    λ = 0.1 * Δ
    ω0 = 0.2 * Δ
    Z = sigmaz()
    X = sigmax()
    Hsys = 0.5 * Δ * Z
    ρ0 = ket2dm(basis(2, 0))
    bath = BosonBath(
        X,
        [0.5 * λ^2, 0.5 * λ^2],
        [-1.0im * ω0 + Γ, 1.0im * ω0 + Γ],
        [0.5im * λ^2, -0.5im * λ^2],
        [-1.0im * ω0 + Γ, 1.0im * ω0 + Γ],
    )

    # compare result with mesolve
    a = qeye(2) ⊗ destroy(tier)
    H = Hsys ⊗ qeye(tier) + λ * tensor(X, qeye(tier)) * (a + a') + ω0 * a' * a
    tlist = LinRange(0, 20 / Δ, 100)
    sol_me = mesolve(H, ρ0 ⊗ ket2dm(basis(tier, 1)), tlist, [sqrt(Γ * 2) * a], e_ops = [a' * a])

    # dynamical field
    input1(p, t) = λ * exp(-1.0im * ω0 * t - Γ * t)
    bath_input1 = BosonDynamicalField(X, η_in = input1)

    input2(p, t) = λ * exp(1.0im * ω0 * t - Γ * t)
    bath_input2 = BosonDynamicalField(X, η_in = input2)

    output1R(p, t) = λ * exp(1.0im * ω0 * (p.tout - t) - Γ * (p.tout - t))
    bath_output_1R = BosonDynamicalField(X, η_out_fn_R = output1R)

    output2L(p, t) = λ * exp(-1.0im * ω0 * (p.tout - t) - Γ * (p.tout - t))
    bath_output_2L = BosonDynamicalField(X, η_out_fn_L = output2L)

    baths = [bath, bath_output_1R, bath_output_2L, bath_input1, bath_input2]

    M = cu(M_Boson(Hsys, tier, baths))
    HDict = M.hierarchy
    e_ops = [
        TrADO(M, 1),
        TrADO(M, HDict.nvec2idx[Nvec([0, 0, 0, 1, 1, 0])]),
        TrADO(M, HDict.nvec2idx[Nvec([0, 0, 1, 0, 0, 1])]),
        TrADO(M, HDict.nvec2idx[Nvec([0, 0, 1, 1, 0, 0])]),
        TrADO(M, HDict.nvec2idx[Nvec([0, 0, 1, 1, 1, 1])]),
    ]

    result = ComplexF64[1.0+0.0im]
    for tout in tlist
        (tout == 0) && continue
        p = (tout = tout,)
        sol_heom = HEOMsolve(M, ρ0, [0, tout], params = p, e_ops = e_ops, verbose = false)
        exp_vals = sol_heom.expect[:, end]
        push!(
            result,
            exp(-2 * Γ * tout) * exp_vals[1] - exp(1im * ω0 * tout - Γ * tout) * exp_vals[2] -
            exp(-1im * ω0 * tout - Γ * tout) * exp_vals[3] - exp_vals[4] + exp_vals[5],
        )
    end
    @test all(isapprox.(result, sol_me.expect[1, :], atol = 1e-6))
end
