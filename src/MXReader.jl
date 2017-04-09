module MXReader

abstract AbstractModel
type FeedForward <: AbstractModel
  arch        :: SymbolicNode
  ctx         :: Vector{Context}

  arg_params  :: Dict{Base.Symbol, NDArray}
  aux_params  :: Dict{Base.Symbol, NDArray}

  pred_exec   :: Union{Executor, Void}
  FeedForward(arch :: SymbolicNode, ctx :: Vector{Context}) = new(arch, ctx)
end

typealias MX_uint Cuint
typealias MX_handle Ptr{Void}
typealias char_p Ptr{UInt8}
typealias char_pp Ptr{char_p}

function _ndarray_alloc()
  h_ref = Ref{MX_handle}(0)
  @mxcall(:MXNDArrayCreateNone, (Ref{MX_handle},), h_ref)
  return MX_NDArrayHandle(h_ref[])
end

function nd_load(fname)
  out_size = Ref{MX_uint}(0)
  handles = Ref{Ptr{MX_handle}}(0)
  out_name_size = Ref{MX_uint}(0)
  names = Ref{char_pp}(0)
  @mxcall(:MXNDArrayLoad, (char_p, Ref{MX_uint}, Ref{Ptr{MX_handle}}, Ref{MX_uint}, Ref{char_pp}),
          filename, out_size, handles, out_name_size, names)
  out_name_size = out_name_size[1]
  out_size      = out_size[1]
  if out_name_size == 0
    return [NDArray(MX_NDArrayHandle(handle)) for handle in unsafe_wrap(Array, handles[], out_size)]
  else
    @assert out_size == out_name_size
    return Dict([(Symbol(unsafe_string(k)), NDArray(MX_NDArrayHandle(handle))) for (k,handle) in
                 zip(unsafe_wrap(Array, out_names[1], out_size), unsafe_wrap(Array, handles[1], out_size))])
end

function sym_load(files)
  dictall=Dict()
  try
      f=open(files)
      dicttxt = readstring(f)
      close(f)
      dictall=JSON.parse(dicttxt)
  end
end

function load_checkpoint(symfile, paramfile)
  symbol = sym_load(symfile)
  save_dict = nd_load(paramfile)
  arg_params = Dict()
  aux_params = Dict()
  for i in save_dict
    pos=search(i[1],':')
    tp=i[1][1:pos-1]
    name=i[1][pos+1:end]
    if tp == "arg"
      arg_params[name] = i[2]
    end
    if tp == "aux"
      aux_params[name] = i[2]
    end
  end
  return symbol, arg_params, aux_params
end



function load(symfile, paramfile, ctx=None)
    symbol, arg_params, aux_params = load_checkpoint(symfile, paramfile)
    return FeedForward(symbol, ctx, arg_params, aux_params)
end

function read(symfile="", paramfile="")
  if (symfile=="")
    print("Enter path for symbol file: ")
    symfile=chomp(readline(STDIN))
  end
  if (paramfile=="")
    print("Enter path for parameter file: ")
    paramfile=chomp(readline(STDIN))
  end
  obj=load(symfile,paramfile)
end


end # module
