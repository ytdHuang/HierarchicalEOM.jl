@testset "CUDA Extension" verbose = true begin
    CUDA.@time @testset "Basic" begin
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
        L_gpu_csc = CUDA.CUSPARSE.CuSparseMatrixCSC(L_cpu)
        L_gpu_csr = CUDA.CUSPARSE.CuSparseMatrixCSR(L_cpu)
        sol_cpu = heomsolve(L_cpu, ψ0, [0, 10]; e_ops = e_ops, progress_bar = Val(false))
        sol_gpu = heomsolve(L_gpu, ψ0, [0, 10]; e_ops = e_ops, progress_bar = Val(false))
        @test L_gpu.data.A isa CUDA.CUSPARSE.CuSparseMatrixCSC
        @test L_gpu_csc.data.A isa CUDA.CUSPARSE.CuSparseMatrixCSC
        @test L_gpu_csr.data.A isa CUDA.CUSPARSE.CuSparseMatrixCSR
        @test all(isapprox.(sol_cpu.expect[1, :], sol_gpu.expect[1, :], atol = 1.0e-4))
        @test isapprox(getRho(sol_cpu.ados[end]), getRho(sol_gpu.ados[end]), atol = 1.0e-4)

        ## Boson HEOMLS
        L_cpu = M_Boson(Hsys, tier, Bbath; verbose = false)
        L_gpu = cu(L_cpu)
        L_cpu_lazy = M_Boson(Hsys, tier, Bbath; verbose = false, assemble = Val(:combine))
        L_gpu_lazy = cu(L_cpu_lazy)
        sol_cpu = heomsolve(L_cpu, ψ0, [0, 10]; e_ops = e_ops, progress_bar = Val(false))
        sol_gpu = heomsolve(L_gpu, ψ0, [0, 10]; e_ops = e_ops, progress_bar = Val(false))
        sol_gpu_lazy = heomsolve(L_gpu_lazy, ψ0, [0, 10]; e_ops = e_ops, progress_bar = Val(false))
        @test L_gpu.data.A isa CUDA.CUSPARSE.CuSparseMatrixCSC{ComplexF64, Int32}
        @test all(isapprox.(sol_cpu.expect[1, :], sol_gpu.expect[1, :], atol = 1.0e-4))
        @test all(isapprox.(sol_cpu.expect[1, :], sol_gpu_lazy.expect[1, :], atol = 1.0e-4))
        @test isapprox(getRho(sol_cpu.ados[end]), getRho(sol_gpu.ados[end]), atol = 1.0e-4)
        @test isapprox(getRho(sol_cpu.ados[end]), getRho(sol_gpu_lazy.ados[end]), atol = 1.0e-4)

        ## Fermion HEOMLS
        L_cpu = M_Fermion(Hsys, tier, Fbath; verbose = false)
        L_cpu_lazy = M_Fermion(Hsys, tier, Fbath; verbose = false, assemble = Val(:combine))
        L_gpu = cu(L_cpu)
        L_gpu_lazy = cu(L_cpu_lazy)
        sol_cpu = heomsolve(L_cpu, ψ0, [0, 10]; e_ops = e_ops, progress_bar = Val(false))
        sol_gpu = heomsolve(L_gpu, ψ0, [0, 10]; e_ops = e_ops, progress_bar = Val(false))
        sol_gpu_lazy = heomsolve(L_gpu_lazy, ψ0, [0, 10]; e_ops = e_ops, progress_bar = Val(false))
        @test L_gpu.data.A isa CUDA.CUSPARSE.CuSparseMatrixCSC{ComplexF64, Int32}
        @test all(isapprox.(sol_cpu.expect[1, :], sol_gpu.expect[1, :], atol = 1.0e-4))
        @test all(isapprox.(sol_cpu.expect[1, :], sol_gpu_lazy.expect[1, :], atol = 1.0e-4))
        @test isapprox(getRho(sol_cpu.ados[end]), getRho(sol_gpu.ados[end]), atol = 1.0e-4)
        @test isapprox(getRho(sol_cpu.ados[end]), getRho(sol_gpu_lazy.ados[end]), atol = 1.0e-4)

        ## Boson Fermion HEOMLS
        L_cpu = M_Boson_Fermion(Hsys, tier, tier, Bbath, Fbath; verbose = false)
        L_cpu_lazy = M_Boson_Fermion(Hsys, tier, tier, Bbath, Fbath; verbose = false, assemble = Val(:combine))
        L_gpu = cu(L_cpu)
        L_gpu_lazy = cu(L_cpu_lazy)
        tlist = 0:1:10
        sol_cpu = heomsolve(L_cpu, ψ0, tlist; e_ops = e_ops, saveat = tlist, progress_bar = Val(false))
        sol_gpu = heomsolve(L_gpu, ψ0, tlist; e_ops = e_ops, saveat = tlist, progress_bar = Val(false))
        sol_gpu_lazy = heomsolve(L_gpu_lazy, ψ0, tlist; e_ops = e_ops, saveat = tlist, progress_bar = Val(false))
        @test L_gpu.data.A isa CUDA.CUSPARSE.CuSparseMatrixCSC{ComplexF64, Int32}
        @test all(isapprox.(sol_cpu.expect[1, :], sol_gpu.expect[1, :], atol = 1.0e-4))
        @test all(isapprox.(sol_cpu.expect[1, :], sol_gpu_lazy.expect[1, :], atol = 1.0e-4))
        for i in 1:length(tlist)
            @test isapprox(getRho(sol_cpu.ados[i]), getRho(sol_gpu.ados[i]), atol = 1.0e-4)
            @test isapprox(getRho(sol_cpu.ados[i]), getRho(sol_gpu_lazy.ados[i]), atol = 1.0e-4)
        end
    end

    CUDA.@time @testset "Single impurity Anderson model" begin
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
        L_even_cpu_lazy = M_Fermion(Hsys, tier, bath_list; verbose = false, assemble = Val(:combine))
        L_even_gpu = cu(L_even_cpu)
        L_even_gpu_lazy = cu(L_even_cpu_lazy)
        ados_cpu = steadystate(L_even_cpu; verbose = false)
        ados_gpu1 = steadystate(L_even_gpu; verbose = false)
        ados_gpu2 = steadystate(CUDA.CUSPARSE.CuSparseMatrixCSR(L_even_cpu); verbose = false)
        ados_gpu3 = steadystate(L_even_gpu, ψ0, 10; verbose = false)
        ados_gpu_lazy = steadystate(L_even_gpu_lazy; verbose = false)
        @test L_even_gpu.data.A isa CUDA.CUSPARSE.CuSparseMatrixCSC{ComplexF64, Int32}
        @test all(isapprox.(ados_cpu.data, ados_gpu1.data; atol = 1.0e-6))
        @test all(isapprox.(ados_cpu.data, ados_gpu2.data; atol = 1.0e-6))
        @test all(isapprox.(ados_cpu.data, ados_gpu3.data; atol = 1.0e-6))
        @test all(isapprox.(ados_cpu.data, ados_gpu_lazy.data; atol = 1.0e-6))

        ## solve density of states
        ωlist = -5:0.5:5
        L_odd_cpu = M_Fermion(Hsys, tier, bath_list, ODD; verbose = false)
        L_odd_cpu_lazy = M_Fermion(Hsys, tier, bath_list, ODD; verbose = false, assemble = Val(:combine))
        L_odd_gpu_32 = cu(L_odd_cpu, word_size = Val(32))
        L_odd_gpu_64 = cu(L_odd_cpu, word_size = Val(64))
        L_odd_gpu_32_lazy = cu(L_odd_cpu_lazy, word_size = Val(32))
        L_odd_gpu_64_lazy = cu(L_odd_cpu_lazy, word_size = Val(64))
        dos_cpu = DensityOfStates(L_odd_cpu, ados_cpu, d_up, ωlist; progress_bar = Val(false))
        dos_gpu_32 = DensityOfStates(
            L_odd_gpu_32,
            ados_cpu,
            d_up,
            ωlist;
            progress_bar = Val(false),
            alg = KrylovJL_BICGSTAB(rtol = 1.0f-12, atol = 1.0f-14), # somehow KrylovJL_GMRES doesn't work for Float32 (it takes forever to solve)
        )
        dos_gpu_64 = DensityOfStates(L_odd_gpu_64, ados_cpu, d_up, ωlist; progress_bar = Val(false))
        dos_gpu_32_lazy = DensityOfStates(
            L_odd_gpu_32_lazy,
            ados_cpu,
            d_up,
            ωlist;
            progress_bar = Val(false),
            alg = KrylovJL_BICGSTAB(rtol = 1.0f-12, atol = 1.0f-14), # somehow KrylovJL_GMRES doesn't work for Float32 (it takes forever to solve)
        )
        dos_gpu_64_lazy = DensityOfStates(L_odd_gpu_64_lazy, ados_cpu, d_up, ωlist; progress_bar = Val(false))
        @test L_odd_gpu_32.data.A isa CUDA.CUSPARSE.CuSparseMatrixCSC{ComplexF32, Int32}
        @test L_odd_gpu_64.data.A isa CUDA.CUSPARSE.CuSparseMatrixCSC{ComplexF64, Int32}
        for (i, ω) in enumerate(ωlist)
            @test dos_cpu[i] ≈ dos_gpu_32[i] atol = 1.0e-6
            @test dos_cpu[i] ≈ dos_gpu_64[i] atol = 1.0e-6
            @test dos_cpu[i] ≈ dos_gpu_32_lazy[i] atol = 1.0e-6
            @test dos_cpu[i] ≈ dos_gpu_64_lazy[i] atol = 1.0e-6
        end
    end

    @testset "IO-HEOM" begin
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

        M_full_cpu = M_Boson(Hsys, tier, baths; assemble = Val(:full), verbose = false)
        M_lazy_cpu = M_Boson(Hsys, tier, baths; assemble = Val(:combine), verbose = false) 
        M_full_gpu = cu(M_full_cpu)
        M_lazy_gpu = cu(M_lazy_cpu)
        HDict = M_full_cpu.hierarchy
        e_ops = [
            TrADO(M_full_cpu, 1),
            TrADO(M_full_cpu, HDict.nvec2idx[Nvec([0, 0, 0, 1, 1, 0])]),
            TrADO(M_full_cpu, HDict.nvec2idx[Nvec([0, 0, 1, 0, 0, 1])]),
            TrADO(M_full_cpu, HDict.nvec2idx[Nvec([0, 0, 1, 1, 0, 0])]),
            TrADO(M_full_cpu, HDict.nvec2idx[Nvec([0, 0, 1, 1, 1, 1])]),
        ]

        result_full = ComplexF64[1.0 + 0.0im]
        result_lazy = ComplexF64[1.0 + 0.0im]
        for tout in tlist
            (tout == 0) && continue
            p = (tout = tout,)
            
            # assemble = Val(:full)
            sol_heom_full = HEOMsolve(M_full_gpu, ρ0, [0, tout], params = p, e_ops = e_ops, progress_bar = Val(false))
            exp_vals_full = sol_heom_full.expect[:, end]
            push!(
                result_full,
                exp(-2 * Γ * tout) * exp_vals_full[1] - exp(1im * ω0 * tout - Γ * tout) * exp_vals_full[2] -
                    exp(-1im * ω0 * tout - Γ * tout) * exp_vals_full[3] - exp_vals_full[4] + exp_vals_full[5],
            )

            # assemble = Val(:combine)
            sol_heom_lazy = HEOMsolve(M_lazy_gpu, ρ0, [0, tout], params = p, e_ops = e_ops, progress_bar = Val(false))
            exp_vals_lazy = sol_heom_lazy.expect[:, end]
            push!(
                result_lazy,
                exp(-2 * Γ * tout) * exp_vals_lazy[1] - exp(1im * ω0 * tout - Γ * tout) * exp_vals_lazy[2] -
                    exp(-1im * ω0 * tout - Γ * tout) * exp_vals_lazy[3] - exp_vals_lazy[4] + exp_vals_lazy[5],
            )
        end
        @test all(isapprox.(result_full, sol_me.expect[1, :], atol = 1.0e-6))
        @test all(isapprox.(result_lazy, sol_me.expect[1, :], atol = 1.0e-6))
    end
end
