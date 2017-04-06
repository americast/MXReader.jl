module MXReader

export read

type FeedForward
  symbol
  ctx
  arg_params
  aux_params
end         #needs to be well-defined

const Cuint = UInt32

int MXNDArrayLoad(const char* fname,
                  mx_uint *out_size,
                  NDArrayHandle** out_arr,
                  mx_uint *out_name_size,
                  const char*** out_names) {
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  ret->ret_vec_str.clear();
  API_BEGIN();
  std::vector<NDArray> data;
  std::vector<std::string> &names = ret->ret_vec_str;
  {
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname, "r")); //called at io.h
    mxnet::NDArray::Load(fi.get(), &data, &names);
  }
  ret->ret_handles.resize(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    NDArray *ptr = new NDArray();
    *ptr = data[i];
    ret->ret_handles[i] = ptr;
  }
  ret->ret_vec_charp.resize(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    ret->ret_vec_charp[i] = names[i].c_str();
  }
  *out_size = static_cast<mx_uint>(data.size());
  *out_arr = dmlc::BeginPtr(ret->ret_handles);
  *out_name_size = static_cast<mx_uint>(names.size());
  *out_names = dmlc::BeginPtr(ret->ret_vec_charp);
  API_END();
}


function nd_load(fname):
    """Load array from file.
    See more details in ``save``.
    Parameters
    ----------
    fname : str
        The filename.
    Returns
    -------
    list of NDArray or dict of str to NDArray
        Loaded data.
    """
    #if not isinstance(fname, string_types):
    #    raise TypeError('fname required to be a string')
    out_size = Cuint
    out_name_size = Cuint
    handles = ctypes.POINTER(NDArrayHandle)()
    names = ctypes.POINTER(ctypes.c_char_p)()
    check_call(_LIB.MXNDArrayLoad(c_str(fname),
                                  ctypes.byref(out_size),
                                  ctypes.byref(handles),
                                  ctypes.byref(out_name_size),
                                  ctypes.byref(names)))
    if out_name_size.value == 0
        return [NDArray(NDArrayHandle(handles[i])) for i in (1:out_size.value)]
    else
        assert out_name_size.value == out_size.value
        return dict(
(py_str(names[i]), NDArray(NDArrayHandle(handles[i]))) for i in (1:out_size.value))
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
