module MXReader

export read

type FeedForward
  symbol
  ctx
  arg_params
  aux_params
end         #needs to be well-defined

function sym_load(files)
  dictall=Dict()
  try
      f=open(files)
      dicttxt = readstring(f)
      close(f)
      dictall=JSON.parse(dicttxt)
  end
end
#=
function sym_load(fname)
  handle = SymbolHandle()
  check_call(_LIB.MXSymbolCreateFromFile(c_str(fname), ctypes.byref(handle)))
  return Symbol(handle)
end
=#

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
