export Nvec

@doc raw"""
    struct Nvec
An object which describes the repetition number of each multi-index ensembles in auxiliary density operators.

The `n_vector` (``\vec{n}``) denotes a set of integers:
```math
\{ n_{1,1}, ..., n_{\alpha, k}, ... \}
```
associated with the ``k``-th exponential-expansion term in the ``\alpha``-th bath.
If ``n_{\alpha, k} = 3`` means that the multi-index ensemble ``\{\alpha, k\}`` appears three times in the multi-index vector of ADOs (see the notations in our paper).

The hierarchy level (``L``) for an `n_vector` is given by ``L=\sum_{\alpha, k} n_{\alpha, k}``

# Fields
- `data` : the `n_vector`
- `level` : The level `L` for the `n_vector`

# Methods
One can obtain the repetition number for specific index (`idx`) by calling : `n_vector[idx]`.
To obtain the corresponding tuple ``(\alpha, k)`` for a given index `idx`, see `bathPtr` in [`HierarchyDict`](@ref) for more details.

`HierarchicalEOM.jl` also supports the following calls (methods) :
```julia
length(n_vector);  # returns the length of `Nvec`
n_vector[1:idx];   # returns a vector which contains the excitation number of `n_vector` from index `1` to `idx`
n_vector[1:end];   # returns a vector which contains all the excitation number of `n_vector`
n_vector[:];       # returns a vector which contains all the excitation number of `n_vector`
from n in n_vector  # iteration
    # do something
end
```
"""
mutable struct Nvec
    data::SparseVector{Int,Int}
    level::Int
end

Nvec(V::Vector{Int}) = Nvec(sparsevec(V), sum(V))
Nvec(V::SparseVector{Int,Int}) = Nvec(copy(V), sum(V))

Base.length(nvec::Nvec) = length(nvec.data)
Base.lastindex(nvec::Nvec) = length(nvec)

Base.getindex(nvec::Nvec, i::T) where {T<:Any} = nvec.data[i]

Base.keys(nvec::Nvec) = keys(nvec.data)

Base.iterate(nvec::Nvec, state::Int = 1) = state > length(nvec) ? nothing : (nvec[state], state + 1)

Base.show(io::IO, nvec::Nvec) = print(io, "Nvec($(nvec[:]))")
Base.show(io::IO, m::MIME"text/plain", nvec::Nvec) = show(io, nvec)

Base.hash(nvec::Nvec, h::UInt) = hash(nvec.data, h)
Base.:(==)(nvec1::Nvec, nvec2::Nvec) = hash(nvec1) == hash(nvec2)

Base.copy(nvec::Nvec) = Nvec(copy(nvec.data), nvec.level)

function Nvec_plus!(nvec::Nvec, idx::Int)
    nvec.data[idx] += 1
    return nvec.level += 1
end
function Nvec_minus!(nvec::Nvec, idx::Int)
    nvec.data[idx] -= 1
    return nvec.level -= 1
end
