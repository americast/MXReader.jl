module MXReader

export load_checkpoint

using MXNet, JSON

function process(symbol, arg_params)
  result=[]
  for i in symbol["nodes"]
    opt=i["op"]
    if opt=="null"
      push!(result,Dict("op"=>opt, "weights"=>copy(arg_params[i["name"]])))
    elseif (opt=="Activation" || opt=="Flatten")
      push!(result,Dict("op"=>opt, "inputs"=>i["inputs"]))
    elseif (opt=="Pooling" || opt=="BatchNorm" || opt=="Convolution"|| opt=="SoftmaxOutput"|| opt=="FullyConnected"||opt=="Concat")
      push!(result,Dict("op"=>opt, "inputs"=>i["inputs"], "param"=>i["param"]))
    else
      push!(result,Dict("op"=>"others",i)
    end
  end
  return result
end
    
    

function readjson(symfile)
 dictall=Dict()
    try
        f=open(joinpath(dirhere, "downloads.json"), "r")
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

end # module
