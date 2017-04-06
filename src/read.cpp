#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <nnvm/c_api.h>
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <nnvm/symbolic.h>
#include "./c_api_common.h"
#include "../operator/operator_common.h"

int MXSymbolCreateFromFile(const char *fname, SymbolHandle *out) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname, "r"));
  dmlc::istream is(fi.get());
  nnvm::Graph g;
  g.attrs["json"] = std::make_shared<nnvm::any>(
    std::string(std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>()));
  s->outputs = nnvm::ApplyPass(g, "LoadLegacyJSON").outputs;
  *out = s;
  is.set_stream(nullptr);
  API_END_HANDLE_ERROR(delete s);
}

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
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname, "r"));
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
