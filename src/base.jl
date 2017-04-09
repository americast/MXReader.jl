include("./loaders.jl")
#from https://github.com/dmlc/MXNet.jl/blob/79d80b6e3f2e66cf42a5e59d0f383ade8e1b4e95/src/base.jl

"Exception thrown when an error occurred calling MXNet API."
immutable MXError <: Exception
  msg :: AbstractString
end

################################################################################
# Initialization and library API entrance
################################################################################

if isempty(MXNET_LIB)
  # touch this file, so that after the user properly build libmxnet, the precompiled
  # MXNet.ji will be re-compiled to get MXNET_LIB properly.
  touch(@__FILE__)
  error("Cannot find or load libmxnet.so. Please see the document on how to build it.")
else
  include_dependency(MXNET_LIB)
end

function __init__()
  # TODO: bug in nnvm, if do not call this, call get handle "_copyto" will fail
  _get_libmx_op_names()
  _populate_iter_creator_cache!()

  atexit() do
    # notify libmxnet we are shutting down
    ccall( ("MXNotifyShutdown", MXNET_LIB), Cint, () )
  end
end


################################################################################
# Handle types
################################################################################
# shifted to "./loaders.jl"
################################################################################
# MXNet Params
#
# MXNet API use string to pass some common parameters like the configurations
# when defining layers. Typically, it is enough to use string(obj) to get a
# recognizable representation for libmxnet. However, there is currently a
# caveat:
#
# Because Julia use column-major ordering for tensors. In order to properly
# interact with Julia Arrays, the shape will look "reversed" from the Julia
# side. For example, a typical MNIST mini-batch tensor is of shape (28,28,1,100)
# from Julia side, while the shape information for the same piece of memory
# should be interpreted as (100,1,28,28) from C/C++/Python side.
#
# Therefore, when passing parameters to libmxnet, we should reverse the shape
# parameter. For example, when the user specify a non-square kernel size for
# a convolution or pooling layer. Unfortunately, those operators are automatically
# imported, and information about the type of each parameter is somehow limited.
# One hacky way is to match the type description for the string "Shape(tuple)"
# when importing operators. But currently we simply decided to reverse **all**
# NTuple{N, Int} passed to libmxnet.
#
# TODO: find a better solution in case this cause issues in the future.
################################################################################
function dump_mx_param(val :: Any)
  string(val)
end
function dump_mx_param{N,T<:Integer}(shape :: NTuple{N, T})
  string(tuple(flipdim([shape...],1)...))
end

"""A convenient macro copied from Mocha.jl that could be used to define structs
with default values and type checks. For example
```julia
@defstruct MyStruct Any (
  field1 :: Int = 0,
  (field2 :: AbstractString = "", !isempty(field2))
)
```
where each field could be either
```julia
field_name :: field_type = default_value
```
or put within a tuple, with the second element
specifying a validation check on the field value.
In the example above, the default value for
field2 does not satisfy the assertion, this
could be used to force user to provide a
valid value when no meaningful default value
is available.

The macro will define a constructor that could accept
the keyword arguments.
"""
macro defstruct(name, fields)
  _defstruct_impl(false, name, fields)
end

"""A convenient macro to define immutable structs. The same as
`@defstruct` except that the defined type is immutable.
"""
macro defimmutable(name, fields)
  _defstruct_impl(true, name, fields)
end

"""Internal use only, this value is used to indicate a required value
is not specified.
"""
immutable __Undefined
end

function _defstruct_impl(is_immutable, name, fields)
  if isa(fields, Expr) && fields.head == :tuple
    fields = fields.args
  else
    fields = [fields]
  end
  @assert length(fields) > 0

  if isa(name, Symbol)
    name       = esc(name)
    super_name = :Any
  elseif VERSION >= v"0.5-"
    @assert(isa(name, Expr) && name.head == :(<:) && length(name.args) == 2 &&
            isa(name.args[1], Symbol) && isa(name.args[2], Symbol),
            "name must be of form 'Name <: SuperType'")

    super_name = esc(name.args[2])
    name       = esc(name.args[1])
  else
    @assert(isa(name, Expr) && name.head == :comparison &&
            length(name.args) == 3 && name.args[2] == :(<:) &&
            isa(name.args[1], Symbol) && isa(name.args[3], Symbol),
            "name must be of form 'Name <: SuperType'")

    super_name = esc(name.args[3])
    name       = esc(name.args[1])
  end

  field_defs     = Vector{Expr}(length(fields))        # :(field2 :: Int)
  field_names    = Vector{Expr}(length(fields))        # :field2
  field_defaults = Vector{Expr}(length(fields))        # :(field2 = 0)
  field_types    = Vector{Expr}(length(fields))        # Int
  field_asserts  = Vector{Expr}(length(fields))        # :(field2 >= 0)
  required_field = Symbol[]

  for i = 1:length(fields)
    field = fields[i]
    if field.head == :tuple
      field_asserts[i] = esc(field.args[2])
      field = field.args[1]
    end
    if field.head == :(=)
      fname             = field.args[1].args[1]
      field_defs[i]     = esc(field.args[1])
      field_names[i]    = esc(fname)
      field_types[i]    = esc(field.args[1].args[2])
      field_defaults[i] = Expr(:kw, fname, esc(field.args[2]))
    else
      # no default value provided, required field
      fname             = field.args[1]
      field_defs[i]     = esc(field)
      field_names[i]    = esc(fname)
      field_types[i]    = esc(field.args[2])
      field_defaults[i] = Expr(:kw, fname, __Undefined())
      push!(required_field, fname)
    end
  end

  # body of layer type, defining fields
  type_body = Expr(:block, field_defs...)

  # constructor
  requires = map(required_field) do fname
    :(@assert(!isa($fname, __Undefined), "value for " * string($fname) * " is required"))
  end
  converts = map(zip(field_names, field_types)) do param
    f_name, f_type = param
    :($f_name = convert($f_type, $f_name))
  end
  asserts = map(filter(i -> isassigned(field_asserts,i), 1:length(fields))) do i
    :(@assert($(field_asserts[i])))
  end
  construct = Expr(:call, name, field_names...)
  ctor_body = Expr(:block, requires..., converts..., asserts..., construct)
  ctor_def = Expr(:call, name, Expr(:parameters, field_defaults...))
  ctor = Expr(:(=), ctor_def, ctor_body)

  if is_immutable
    quote
      immutable $(name) <: $(super_name)
        $type_body
      end

      $ctor
    end
  else
    quote
      type $(name) <: $(super_name)
        $type_body
      end

      $ctor
    end
  end
end
