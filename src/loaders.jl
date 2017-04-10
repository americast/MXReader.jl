const MXNET_LIB = Libdl.find_library(["libmxnet.so","libmxnet.dll"],
                                     [joinpath("$(get(ENV,"MXNET_HOME",""))","lib"),
                                      Pkg.dir("MXReader","deps","usr","lib")])

################################################################################
# Common types used in MXNet API
################################################################################
typealias MX_uint Cuint
typealias MX_float Cfloat
typealias MX_handle Ptr{Void}

typealias char_p Ptr{UInt8}
typealias char_pp Ptr{char_p}

"Utility macro to call MXNet API functions"
macro mxcall(fv, argtypes, args...)
  f = eval(fv)
  args = map(esc, args)
  quote
    _mxret = ccall( ($(Meta.quot(f)), $MXNET_LIB),
                    Cint, $argtypes, $(args...) )
    if _mxret != 0
      err_msg = mx_get_last_error()
      throw(MXError(err_msg))
    end
  end
end

macro mx_define_handle_t(name, destructor)
  name = esc(name)
  quote
    type $name
      value :: MX_handle

      function $name(value = C_NULL)
        hdr = new(value)

        $(if destructor != :nop
          :(finalizer(hdr, delete!))
        end)

        return hdr
      end
    end

    $(if finalizer != :nop
      quote
        function delete!(h :: $name)
          if h.value != C_NULL
            @mxcall($(Meta.quot(destructor)), (MX_handle,), h.value)
            h.value = C_NULL
          end
        end
      end
    end)

    function Base.unsafe_convert(::Type{MX_handle}, obj::$name)
      obj.value
    end
    Base.convert(t::Type{MX_handle}, obj::$name) = Base.unsafe_convert(t, obj)
    Base.cconvert(t::Type{MX_handle}, obj::$name) = Base.unsafe_convert(t, obj)

    function Base.isnull(obj::$name) obj.value == C_NULL end
  end
end

@mx_define_handle_t(MX_NDArrayHandle, MXNDArrayFree)
@mx_define_handle_t(MX_OpHandle, nop)
@mx_define_handle_t(MX_SymbolHandle, MXSymbolFree)
@mx_define_handle_t(MX_ExecutorHandle, MXExecutorFree)
@mx_define_handle_t(MX_DataIterHandle, MXDataIterFree)
@mx_define_handle_t(MX_KVStoreHandle, MXKVStoreFree)

function mx_get_last_error()
  msg = ccall( ("MXGetLastError", MXNET_LIB), char_p, () )
  if msg == C_NULL
    throw(MXError("Failed to get last error message"))
  end
  return unsafe_string(msg)
end

function _get_iter_creators()
  n_ref = Ref{MX_uint}(0)
  h_ref = Ref{Ptr{MX_handle}}(0)
  @mxcall(:MXListDataIters, (Ref{MX_uint}, Ref{Ptr{MX_handle}}), n_ref, h_ref)

  return unsafe_wrap(Array, h_ref[], n_ref[])
end

function _get_iter_name(hdr :: MX_handle)
  ref_name      = Ref{char_p}(0)
  ref_desc      = Ref{char_p}(0)
  ref_narg      = Ref{MX_uint}(0)
  ref_arg_names = Ref{char_pp}(0)
  ref_arg_types = Ref{char_pp}(0)
  ref_arg_descs = Ref{char_pp}(0)

  @mxcall(:MXDataIterGetIterInfo,
          (MX_handle, Ref{char_p}, Ref{char_p}, Ref{MX_uint}, Ref{char_pp}, Ref{char_pp}, Ref{char_pp}),
          hdr, ref_name, ref_desc, ref_narg, ref_arg_names, ref_arg_types, ref_arg_descs)

  return Symbol(unsafe_string(ref_name[]))
end

const _iter_creator_cache = Dict{Symbol, MX_handle}()
function _populate_iter_creator_cache!()
  empty!(_iter_creator_cache)
  h_creators = _get_iter_creators()
  for handle in h_creators
    name = _get_iter_name(handle)
    _iter_creator_cache[name] = handle
  end
end


function _get_libmx_op_names()
  n = Ref{MX_uint}(0)
  names = Ref{char_pp}(0)

  @mxcall(:MXListAllOpNames, (Ref{MX_uint}, Ref{char_pp}), n, names)

  names = unsafe_wrap(Array, names[], n[])
  return [unsafe_string(x) for x in names]
end

function sym_load(filename::String)
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXSymbolCreateFromFile, (char_p, Ref{MX_handle}), filename, ref_hdr)
  return SymbolicNode(MX_SymbolHandle(ref_hdr[]))
end


type NDArray
  handle   :: MX_NDArrayHandle
  writable :: Bool

  function NDArray(handle, writable=true)
    new(handle, writable)
  end
end


type SymbolicNode
  handle :: MX_SymbolHandle
end

@enum CONTEXT_TYPE CPU=1 GPU=2 CPU_PINNED=3

immutable Context
  device_type :: CONTEXT_TYPE
  device_id   :: Int
end
Context(dev_type :: Union{CONTEXT_TYPE, Int}, dev_id :: Int = 0) =
Context(convert(CONTEXT_TYPE, dev_type), dev_id)



function nd_load(filename::AbstractString)
  out_size      = Ref{MX_uint}(0)
  out_hdrs      = Ref{Ptr{MX_handle}}(0)
  out_name_size = Ref{MX_uint}(0)
  out_names     = Ref{char_pp}(0)
  @mxcall(:MXNDArrayLoad, (char_p, Ref{MX_uint}, Ref{Ptr{MX_handle}}, Ref{MX_uint}, Ref{char_pp}),
          filename, out_size, out_hdrs, out_name_size, out_names)
  out_name_size = out_name_size[]
  out_size      = out_size[]
  if out_name_size == 0
    return [NDArray(MX_NDArrayHandle(hdr)) for hdr in unsafe_wrap(Array, out_hdrs[], out_size)]
  else
    @assert out_size == out_name_size
    return Dict([(Symbol(unsafe_string(k)), NDArray(MX_NDArrayHandle(hdr))) for (k,hdr) in
                 zip(unsafe_wrap(Array, out_names[], out_size), unsafe_wrap(Array, out_hdrs[], out_size))])
  end
end

type Executor
  handle :: MX_ExecutorHandle
  symbol :: SymbolicNode
  arg_arrays  :: Vector{NDArray}
  grad_arrays :: Vector{Union{Void,NDArray}}
  aux_arrays  :: Vector{NDArray}
  outputs     :: Vector{NDArray}
  arg_dict    :: Dict{Base.Symbol, NDArray}
  aux_dict    :: Dict{Base.Symbol, NDArray}
end
