module MXReader

include("./model.jl")
export readf

function load_checkpoint(symfile, paramfile)
  print(typeof(symfile))
  symbol = sym_load(symfile)
  save_dict = nd_load(paramfile)
  arg_params = Dict()
  aux_params = Dict()
  for i in save_dict
    pos=search("$(i[1])",':')
    tp="$(i[1])"[1:pos-1]
    name="$(i[1])"[pos+1:end]
    if tp == "arg"
      arg_params[name] = i[2]
    end
    if tp == "aux"
      aux_params[name] = i[2]
    end
  end
  return symbol, arg_params, aux_params
end



function load(symfile, paramfile)
    symbol, arg_params, aux_params = load_checkpoint(symfile, paramfile)
    return Feed(symbol, arg_params, aux_params)
end

function readf(symfile="", paramfile="", out=1)
  if (symfile=="")
    print("Enter path for symbol file: ")
    symfile=chomp(readline(STDIN))
  end
  if (paramfile=="")
    print("Enter path for parameter file: ")
    paramfile=chomp(readline(STDIN))
  end
  obj = load(symfile,paramfile)
  if (out==1)
    return obj
  else
    println(obj)
  end
end

end # module
