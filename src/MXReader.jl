module MXReader

export readf

using MXNet, JSON

function process(symbol, arg_params)
  result=[]
  for i in symbol["nodes"]
    opt=i["op"]
    if opt=="null"
      try
        push!(result,Dict("op"=>opt, "weights"=>copy(arg_params[i["name"]])))
      catch
        warn("Place input data as first entry")
      end
    elseif (opt=="Activation" || opt=="Flatten")
      push!(result,Dict("op"=>opt, "inputs"=>i["inputs"]))
    elseif (opt=="Pooling" || opt=="BatchNorm" || opt=="Convolution"|| opt=="SoftmaxOutput"|| opt=="FullyConnected"||opt=="Concat")
      push!(result,Dict("op"=>opt, "inputs"=>i["inputs"], "param"=>i["param"]))
    else
      push!(result,Dict("op"=>"others",i))
    end
  end
  return result
end
    
    

function readjson(symfile)
 dictall=Dict()
    try
        f=open(symfile, "r")
        dicttxt = readstring(f)
        close(f)
        dictall=JSON.parse(dicttxt)
    end
  return dictall
end

function load_checkpoint(symfile, paramfile)
  symbol = readjson(symfile)
  save_dict = mx.load(paramfile,mx.NDArray)
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
  #return symbol,arg_params, aux_params
  process(symbol,arg_params)
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
  res = load_checkpoint(symfile,paramfile)
  if (out==1)
    return res
  else
    println(res)
  end
end

end # module
