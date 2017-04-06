#include<bits/stdc++.h>
#include"./io.h"
#include"./base.h"
#include"./c_api.h"
#include"./thread_local.h"

struct MXAPIThreadLocalEntry {
  /*! \brief result holder for returning string */
  std::string ret_str;
  /*! \brief result holder for returning strings */
  std::vector<std::string> ret_vec_str;
  /*! \brief result holder for returning string pointers */
  std::vector<const char *> ret_vec_charp;
  /*! \brief result holder for returning handles */
  std::vector<void *> ret_handles;
  /*! \brief result holder for returning shapes */
  //std::vector<TShape> arg_shapes, out_shapes, aux_shapes;
  /*! \brief result holder for returning type flags */
  std::vector<int> arg_types, out_types, aux_types;
  /*! \brief result holder for returning shape dimensions */
  std::vector<mx_uint> arg_shape_ndim, out_shape_ndim, aux_shape_ndim;
  /*! \brief result holder for returning shape pointer */
  std::vector<const mx_uint*> arg_shape_data, out_shape_data, aux_shape_data;
  // helper function to setup return value of shape array
  /*inline static void SetupShapeArrayReturn(
      const std::vector<TShape> &shapes,
      std::vector<mx_uint> *ndim,
      std::vector<const mx_uint*> *data) {
    ndim->resize(shapes.size());
    data->resize(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
      ndim->at(i) = shapes[i].ndim();
      data->at(i) = shapes[i].data();
    }
  }*/
};
typedef dmlc::ThreadLocalStore<MXAPIThreadLocalEntry> MXAPIThreadLocalStore;

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
