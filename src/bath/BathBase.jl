export AbstractBath
export BathTerm, Exponent, DynamicalField
export correlation_function

abstract type AbstractBath end
abstract type BathTerm end

@doc raw"""
    struct Exponent <: BathTerm
An object which describes a single exponential-expansion term (naively, an excitation mode) within the decomposition of the bath correlation functions.

The expansion of a bath correlation function can be expressed as : ``C(t) = \sum_i \eta_i \exp(-\gamma_i t)``.

# Fields
- `op::QuantumObject` : The system coupling operator according to system-bath interaction.
- `η::Number` : the coefficient ``\eta_i`` in bath correlation function.
- `γ::Number` : the coefficient ``\gamma_i`` in bath correlation function.
- `types::String` : The type-tag of the exponent.

The different types of the Exponent:
- `\"bR\"` : from real part of bosonic correlation function ``C^{u=\textrm{R}}(t)``
- `\"bI\"` : from imaginary part of bosonic correlation function ``C^{u=\textrm{I}}(t)``
- `\"bRI\"` : from combined (real and imaginary part) bosonic bath correlation function ``C(t)``
- `\"bA\"` : from absorption bosonic correlation function ``C^{\nu=+}(t)``
- `\"bE\"` : from emission bosonic correlation function ``C^{\nu=-}(t)``
- `\"fA\"` : from absorption fermionic correlation function ``C^{\nu=+}(t)``
- `\"fE\"` : from emission fermionic correlation function ``C^{\nu=-}(t)``
"""
struct Exponent <: BathTerm
    op::QuantumObject
    η::Number
    γ::Number
    types::String
end

Base.show(io::IO, E::Exponent) = print(io, "Bath Exponent with types = \"$(E.types)\", η = $(E.η), γ = $(E.γ).\n")
Base.show(io::IO, m::MIME"text/plain", E::Exponent) = show(io, E)

@doc raw"""
    struct DynamicalField <: BathTerm
An object which describes a single dynamical field term, which is a correlation function between the system-bath coupling operator and a desired bath operator.

# Fields
- `op::QuantumObject` : The system coupling operator according to system-bath interaction.
- `η::Union{Number,Function}` : the coefficient ``\eta_i``. If `η(p, t)` is a `Function`, it must depends on the parameters `p` and time `t`
- `γ::Number` : the coefficient ``\gamma_i``.
- `types::String` : The type-tag of the dynamical field.

The different types of the DynamicalField:
- `\"bInFn\"`
- `\"bOutFnL\"`
- `\"bOutFnR\"`
- `\"bOutL\"`
- `\"bOutR\"`
"""
struct DynamicalField <: BathTerm
    op::QuantumObject
    η::Union{Number,ScalarOperator}
    γ::Number
    types::String
end

Base.show(io::IO, F::DynamicalField) =
    print(io, "Dynamical Field with types = \"$(F.types)\", η = $(F.η), γ = $(F.γ).\n")
Base.show(io::IO, m::MIME"text/plain", F::DynamicalField) = show(io, F)

Base.show(io::IO, B::AbstractBath) = print(io, "$(typeof(B)) object with $(B.Nterm) terms.\n")
Base.show(io::IO, m::MIME"text/plain", B::AbstractBath) = show(io, B)

Base.checkbounds(B::AbstractBath, i::Int) =
    ((i < 1) || (i > B.Nterm)) ? error("Attempt to access $(B.Nterm)-term Bath at index [$(i)]") : nothing

Base.length(B::AbstractBath) = B.Nterm
Base.lastindex(B::AbstractBath) = B.Nterm

function Base.getindex(B::AbstractBath, i::Int)
    checkbounds(B, i)

    count = 0
    for b in B.bath
        if i <= (count + b.Nterm)
            k = i - count
            b_type = typeof(b)
            if b_type <: AbstractBosonDynamicalField
                if b_type == bosonInputFunction
                    types = "bInFn"
                elseif b_type == bosonOutputFunctionLeft
                    types = "bOutFnL"
                elseif b_type == bosonOutputFunctionRight
                    types = "bOutFnR"
                elseif b_type == bosonOutputLeft
                    types = "bOutL"
                elseif b_type == bosonOutputRight
                    types = "bOutR"
                end
                return DynamicalField(B.op, b.η[k], b.γ[k], types)
            else
                if b_type == bosonRealImag
                    η = b.η_real[k] + 1.0im * b.η_imag[k]
                    types = "bRI"
                    op = B.op
                else
                    η = b.η[k]
                    if b_type == bosonReal
                        types = "bR"
                        op = B.op
                    elseif b_type == bosonImag
                        types = "bI"
                        op = B.op
                    elseif b_type == bosonAbsorb
                        types = "bA"
                        op = B.op'
                    elseif b_type == bosonEmit
                        types = "bE"
                        op = B.op
                    elseif b_type == fermionAbsorb
                        types = "fA"
                        op = B.op'
                    elseif b_type == fermionEmit
                        types = "fE"
                        op = B.op
                    end
                end
                return Exponent(op, η, b.γ[k], types)
            end
        else
            count += b.Nterm
        end
    end
end

function Base.getindex(B::AbstractBath, r::UnitRange{Int})
    checkbounds(B, r[1])
    checkbounds(B, r[end])

    count = 0
    list = BathTerm[]
    for b in B.bath
        for k in 1:b.Nterm
            count += 1
            if (r[1] <= count) && (count <= r[end])
                b_type = typeof(b)
                if b_type <: AbstractBosonDynamicalField
                    if b_type == bosonInputFunction
                        types = "bInFn"
                    elseif b_type == bosonOutputFunctionLeft
                        types = "bOutFnL"
                    elseif b_type == bosonOutputFunctionRight
                        types = "bOutFnR"
                    elseif b_type == bosonOutputLeft
                        types = "bOutL"
                    elseif b_type == bosonOutputRight
                        types = "bOutR"
                    end
                    push!(list, DynamicalField(B.op, b.η[k], b.γ[k], types))
                else
                    if b_type == bosonRealImag
                        η = b.η_real[k] + 1.0im * b.η_imag[k]
                        types = "bRI"
                        op = B.op
                    else
                        η = b.η[k]
                        if b_type == bosonReal
                            types = "bR"
                            op = B.op
                        elseif b_type == bosonImag
                            types = "bI"
                            op = B.op
                        elseif b_type == bosonAbsorb
                            types = "bA"
                            op = B.op'
                        elseif b_type == bosonEmit
                            types = "bE"
                            op = B.op
                        elseif b_type == fermionAbsorb
                            types = "fA"
                            op = B.op'
                        elseif b_type == fermionEmit
                            types = "fE"
                            op = B.op
                        end
                    end
                    push!(list, Exponent(op, η, b.γ[k], types))
                end
            end
            if count == r[end]
                return list
            end
        end
    end
end

Base.getindex(B::AbstractBath, ::Colon) = getindex(B, 1:B.Nterm)

Base.iterate(B::AbstractBath, state::Int = 1) = state > length(B) ? nothing : (B[state], state + 1)

function _check_gamma_absorb_and_emit(γ_absorb, γ_emit)
    len = length(γ_absorb)
    if length(γ_emit) == len
        for k in 1:len
            if !(γ_absorb[k] ≈ conj(γ_emit[k]))
                @warn "The elements in \'γ_absorb\' should be complex conjugate of the corresponding elements in \'γ_emit\'."
            end
        end
    else
        error("The length of \'γ_absorb\' and \'γ_emit\' should be the same.")
    end
end

correlation_function(bath::AbstractBath, t::Real) = correlation_function(bath, [t])
