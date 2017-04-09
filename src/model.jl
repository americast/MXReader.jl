include("./base.jl")

abstract AbstractModel
type FeedForward <: AbstractModel
  arch        :: SymbolicNode
  arg_params  :: Dict{Base.Symbol, NDArray}
  aux_params  :: Dict{Base.Symbol, NDArray}
  ctx         :: Vector{Context}
end

function Feed(arch :: SymbolicNode, arg::Dict{Any,Any}, aux::Dict{Any,Any}, context :: Union{Context, Vector{Context}, Void} = nothing)
  if isa(context, Void)
    context = [Context(CPU)]
  elseif isa(context, Context)
    context = [context]
  end
  return FeedForward(arch, arg, aux, context)
end
