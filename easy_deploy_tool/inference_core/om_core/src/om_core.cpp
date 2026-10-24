#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "om_blob_buffer.hpp"
#include "om_core/om_core.hpp"

namespace easy_deploy {

namespace {

std::mutex g_acl_env_mu;
int        g_acl_env_ref = 0;

void AcquireAclEnv()
{
  std::lock_guard<std::mutex> lock(g_acl_env_mu);
  if (g_acl_env_ref == 0)
  {
    aclError ret = aclInit("");
    CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] aclInit failed, err: %d", ret);
  }
  ++g_acl_env_ref;
}

void ReleaseAclEnv()
{
  std::lock_guard<std::mutex> lock(g_acl_env_mu);
  if (g_acl_env_ref <= 0)
  {
    return;
  }
  --g_acl_env_ref;
  if (g_acl_env_ref == 0)
  {
    aclFinalize();
  }
}

std::vector<size_t> ToSizeShape(const std::vector<uint64_t> &shape)
{
  return std::vector<size_t>(shape.begin(), shape.end());
}

} // namespace

class OmInferCore : public BaseInferCore {
public:
  ~OmInferCore() override;

  OmInferCore(const std::string                                             model_path,
              const std::unordered_map<std::string, std::vector<uint64_t>> &input_blobs_shape,
              const std::unordered_map<std::string, std::vector<uint64_t>> &output_blobs_shape,
              const int                                                     device_id = 0);

  OmInferCore(const std::string model_path, const int device_id = 0);

  std::unique_ptr<BlobsTensor> AllocBlobsBuffer() override;

  InferCoreType GetType()
  {
    return InferCoreType::OM;
  }

  std::string GetName()
  {
    return "om_core";
  }

private:
  bool PreProcess(std::shared_ptr<IPipelinePackage> buffer) override;

  bool Inference(std::shared_ptr<IPipelinePackage> buffer) override;

  bool PostProcess(std::shared_ptr<IPipelinePackage> buffer) override;

  std::unordered_map<std::string, std::vector<uint64_t>> ResolveModelInputInformation();

  std::unordered_map<std::string, std::vector<uint64_t>> ResolveModelOutputInformation();

  std::string ShapeToString(const std::vector<uint64_t> &shape);

  void InitAclRuntime();

  void CreateIoDatasets();

  void ReleaseAclResources();

  void InitTensorTypeByteSize();

  void BuildNameIndexMaps();

  void FillBlobDtypes();

  size_t FindInputIndexByName(const std::string &blob_name) const;

  size_t FindOutputIndexByName(const std::string &blob_name) const;

  void DestroyDataset(aclmdlDataset *&dataset, std::vector<void *> &device_buffers);

private:
  int32_t        device_id_;
  aclrtContext   context_    = nullptr;
  aclrtStream    stream_     = nullptr;
  uint32_t       model_id_   = 0;
  aclmdlDesc    *model_desc_ = nullptr;
  aclrtRunMode   run_mode_;
  bool           acl_env_acquired_ = false;

  std::string                                          model_path_;
  std::unordered_map<aclDataType, size_t>              map_tensor_type_byte_size_;
  std::unordered_map<std::string, std::vector<uint64_t>> map_input_blob_name2shape_;
  std::unordered_map<std::string, std::vector<uint64_t>> map_output_blob_name2shape_;
  std::unordered_map<std::string, aclDataType>         map_blob_name2type_;
  std::unordered_map<std::string, size_t>              map_input_name2index_;
  std::unordered_map<std::string, size_t>              map_output_name2index_;

  aclmdlDataset         *input_dataset_  = nullptr;
  aclmdlDataset         *output_dataset_ = nullptr;
  std::vector<void *>    input_device_buffers_;
  std::vector<void *>    output_device_buffers_;
};

OmInferCore::OmInferCore(const std::string model_path, const int device_id)
    : OmInferCore(model_path, {}, {}, device_id)
{}

OmInferCore::OmInferCore(
    const std::string                                             model_path,
    const std::unordered_map<std::string, std::vector<uint64_t>> &input_blobs_shape,
    const std::unordered_map<std::string, std::vector<uint64_t>> &output_blobs_shape,
    const int                                                     device_id)
{
  model_path_ = model_path;
  device_id_  = device_id;

  try
  {
    InitAclRuntime();

    map_input_blob_name2shape_ =
        input_blobs_shape.empty() ? ResolveModelInputInformation() : input_blobs_shape;
    map_output_blob_name2shape_ =
        output_blobs_shape.empty() ? ResolveModelOutputInformation() : output_blobs_shape;

    InitTensorTypeByteSize();
    BuildNameIndexMaps();
    FillBlobDtypes();
    CreateIoDatasets();
    BaseInferCore::Init();
  } catch (...)
  {
    ReleaseAclResources();
    throw;
  }

  auto func_display_blobs_info =
      [](const std::unordered_map<std::string, std::vector<uint64_t>> &blobs_shape) {
        for (const auto &p_name_shape : blobs_shape)
        {
          std::string s_blob_shape;
          for (const auto dim : p_name_shape.second)
          {
            s_blob_shape += std::to_string(dim) + "\t";
          }
          LOG_DEBUG("blob name : %s, blob shape : %s", p_name_shape.first.c_str(),
                    s_blob_shape.c_str());
        }
      };
  func_display_blobs_info(map_input_blob_name2shape_);
  func_display_blobs_info(map_output_blob_name2shape_);
}

OmInferCore::~OmInferCore()
{
  ReleaseAclResources();
}

std::string OmInferCore::ShapeToString(const std::vector<uint64_t> &shape)
{
  std::stringstream ss;
  for (size_t i = 0; i < shape.size(); ++i)
  {
    if (i > 0)
    {
      ss << ",";
    }
    ss << shape[i];
  }
  return ss.str();
}

void OmInferCore::InitAclRuntime()
{
  LOG_DEBUG("Initializing ACL core with model {%s} ...", model_path_.c_str());
  AcquireAclEnv();
  acl_env_acquired_ = true;

  aclError ret = aclrtSetDevice(device_id_);
  CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] aclrtSetDevice failed, err: %d", ret);

  ret = aclrtCreateContext(&context_, device_id_);
  CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] aclrtCreateContext failed, err: %d", ret);

  ret = aclrtCreateStream(&stream_);
  CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] aclrtCreateStream failed, err: %d", ret);

  ret = aclrtGetRunMode(&run_mode_);
  CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] aclrtGetRunMode failed, err: %d", ret);

  ret = aclmdlLoadFromFile(model_path_.c_str(), &model_id_);
  CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] aclmdlLoadFromFile failed, err: %d", ret);

  model_desc_ = aclmdlCreateDesc();
  ret         = aclmdlGetDesc(model_desc_, model_id_);
  CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] aclmdlGetDesc failed, err: %d", ret);
}

void OmInferCore::CreateIoDatasets()
{
  const size_t input_count = aclmdlGetNumInputs(model_desc_);
  CHECK_STATE_THROW(input_count > 0, "[ACL] No input blobs found");
  CHECK_STATE_THROW(map_input_blob_name2shape_.size() == input_count,
                    "[ACL] Input blob count mismatch: map=%zu, model=%zu",
                    map_input_blob_name2shape_.size(), input_count);

  input_dataset_ = aclmdlCreateDataset();
  CHECK_STATE_THROW(input_dataset_ != nullptr, "[ACL] Create input dataset failed");
  for (size_t i = 0; i < input_count; ++i)
  {
    const size_t buffer_size = aclmdlGetInputSizeByIndex(model_desc_, i);
    void        *device_ptr  = nullptr;
    aclError     ret = aclrtMalloc(&device_ptr, buffer_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] Malloc input[%zu] device buffer failed, err: %d",
                      i, ret);
    input_device_buffers_.push_back(device_ptr);

    aclDataBuffer *data_buffer = aclCreateDataBuffer(device_ptr, buffer_size);
    CHECK_STATE_THROW(data_buffer != nullptr, "[ACL] Create input[%zu] data buffer failed", i);
    ret = aclmdlAddDatasetBuffer(input_dataset_, data_buffer);
    CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] Add input[%zu] dataset buffer failed, err: %d", i,
                      ret);
  }

  const size_t output_count = aclmdlGetNumOutputs(model_desc_);
  CHECK_STATE_THROW(output_count > 0, "[ACL] No output blobs found");
  CHECK_STATE_THROW(map_output_blob_name2shape_.size() == output_count,
                    "[ACL] Output blob count mismatch: map=%zu, model=%zu",
                    map_output_blob_name2shape_.size(), output_count);

  output_dataset_ = aclmdlCreateDataset();
  CHECK_STATE_THROW(output_dataset_ != nullptr, "[ACL] Create output dataset failed");
  for (size_t i = 0; i < output_count; ++i)
  {
    const size_t buffer_size = aclmdlGetOutputSizeByIndex(model_desc_, i);
    void        *device_ptr  = nullptr;
    aclError     ret = aclrtMalloc(&device_ptr, buffer_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] Malloc output[%zu] device buffer failed, err: %d",
                      i, ret);
    output_device_buffers_.push_back(device_ptr);

    aclDataBuffer *data_buffer = aclCreateDataBuffer(device_ptr, buffer_size);
    CHECK_STATE_THROW(data_buffer != nullptr, "[ACL] Create output[%zu] data buffer failed", i);
    ret = aclmdlAddDatasetBuffer(output_dataset_, data_buffer);
    CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] Add output[%zu] dataset buffer failed, err: %d",
                      i, ret);
  }

  LOG_DEBUG("ACL core initialized successfully, inputs=%zu, outputs=%zu", input_count,
            output_count);
}

std::unordered_map<std::string, std::vector<uint64_t>> OmInferCore::ResolveModelInputInformation()
{
  std::unordered_map<std::string, std::vector<uint64_t>> input_shape_map;
  const size_t input_count = aclmdlGetNumInputs(model_desc_);
  CHECK_STATE_THROW(input_count > 0, "[ACL] No input blobs found");

  for (size_t input_idx = 0; input_idx < input_count; ++input_idx)
  {
    const char *raw_input_name = aclmdlGetInputNameByIndex(model_desc_, input_idx);
    CHECK_STATE_THROW(raw_input_name != nullptr && raw_input_name[0] != '\0',
                      "[ACL] Failed to get input name for index %zu", input_idx);
    const std::string blob_name(raw_input_name);

    aclmdlIODims input_dims{};
    aclError     ret_acl = aclmdlGetInputDims(model_desc_, input_idx, &input_dims);
    CHECK_STATE_THROW(ret_acl == ACL_SUCCESS, "[ACL] Get input dims failed, err: %d", ret_acl);

    std::vector<uint64_t> input_shape;
    for (size_t dim_idx = 0; dim_idx < input_dims.dimCount; ++dim_idx)
    {
      int64_t dim_val = input_dims.dims[dim_idx];
      CHECK_STATE_THROW(dim_val > 0, "[ACL] Invalid input dim %lld (must be >0)",
                        static_cast<long long>(dim_val));
      input_shape.push_back(static_cast<uint64_t>(dim_val));
    }

    input_shape_map[blob_name] = input_shape;
    LOG_DEBUG("[ACL] Resolved input tensor: name=%s, shape=[%s]", blob_name.c_str(),
              ShapeToString(input_shape).c_str());
  }

  CHECK_STATE_THROW(!input_shape_map.empty(), "[ACL] Resolved empty input shape map (bug!)");
  return input_shape_map;
}

std::unordered_map<std::string, std::vector<uint64_t>> OmInferCore::ResolveModelOutputInformation()
{
  std::unordered_map<std::string, std::vector<uint64_t>> output_shape_map;
  const size_t output_count = aclmdlGetNumOutputs(model_desc_);
  CHECK_STATE_THROW(output_count > 0, "[ACL] No output blobs found");

  for (size_t output_idx = 0; output_idx < output_count; ++output_idx)
  {
    const char *raw_output_name = aclmdlGetOutputNameByIndex(model_desc_, output_idx);
    CHECK_STATE_THROW(raw_output_name != nullptr && raw_output_name[0] != '\0',
                      "[ACL] Failed to get output name for index %zu", output_idx);
    const std::string blob_name(raw_output_name);

    aclmdlIODims output_dims{};
    aclError     acl_ret = aclmdlGetOutputDims(model_desc_, output_idx, &output_dims);
    CHECK_STATE_THROW(acl_ret == ACL_SUCCESS,
                      "[ACL] aclmdlGetOutputDims failed for output[%zu], err_code: %d", output_idx,
                      acl_ret);
    CHECK_STATE_THROW(output_dims.dimCount > 0 && output_dims.dimCount <= ACL_MAX_DIM_CNT,
                      "[ACL] Invalid dim count for output[%zu]: dimCount = %zu (max: %d)",
                      output_idx, output_dims.dimCount, ACL_MAX_DIM_CNT);

    std::vector<uint64_t> output_shape;
    for (size_t dim_idx = 0; dim_idx < output_dims.dimCount; ++dim_idx)
    {
      int64_t dim_val = output_dims.dims[dim_idx];
      CHECK_STATE_THROW(dim_val > 0, "[ACL] Invalid dim for output[%zu][dim%zu]: %lld", output_idx,
                        dim_idx, static_cast<long long>(dim_val));
      output_shape.push_back(static_cast<uint64_t>(dim_val));
    }

    output_shape_map[blob_name] = output_shape;
    LOG_DEBUG("[ACL] Resolved output tensor: name=%s, shape=[%s]", blob_name.c_str(),
              ShapeToString(output_shape).c_str());
  }

  CHECK_STATE_THROW(!output_shape_map.empty(), "[ACL] Resolved empty output shape map (bug!)");
  return output_shape_map;
}

void OmInferCore::InitTensorTypeByteSize()
{
  map_tensor_type_byte_size_ = {
      {ACL_FLOAT, 4}, {ACL_INT32, 4},  {ACL_UINT8, 1}, {ACL_FLOAT16, 2},
      {ACL_INT8, 1},  {ACL_INT64, 8},  {ACL_UINT64, 8}, {ACL_UINT32, 4}};
}

size_t OmInferCore::FindInputIndexByName(const std::string &blob_name) const
{
  const size_t input_count = aclmdlGetNumInputs(model_desc_);
  for (size_t i = 0; i < input_count; ++i)
  {
    const char *raw_name = aclmdlGetInputNameByIndex(model_desc_, i);
    if (raw_name != nullptr && blob_name == raw_name)
    {
      return i;
    }
  }
  CHECK_STATE_THROW(false, "[ACL] Input blob name not found in OM: %s", blob_name.c_str());
  return 0;
}

size_t OmInferCore::FindOutputIndexByName(const std::string &blob_name) const
{
  const size_t output_count = aclmdlGetNumOutputs(model_desc_);
  for (size_t i = 0; i < output_count; ++i)
  {
    const char *raw_name = aclmdlGetOutputNameByIndex(model_desc_, i);
    if (raw_name != nullptr && blob_name == raw_name)
    {
      return i;
    }
  }
  CHECK_STATE_THROW(false, "[ACL] Output blob name not found in OM: %s", blob_name.c_str());
  return 0;
}

void OmInferCore::BuildNameIndexMaps()
{
  map_input_name2index_.clear();
  map_output_name2index_.clear();
  for (const auto &p_name_shape : map_input_blob_name2shape_)
  {
    map_input_name2index_[p_name_shape.first] = FindInputIndexByName(p_name_shape.first);
  }
  for (const auto &p_name_shape : map_output_blob_name2shape_)
  {
    map_output_name2index_[p_name_shape.first] = FindOutputIndexByName(p_name_shape.first);
  }
}

void OmInferCore::FillBlobDtypes()
{
  for (const auto &p_name_shape : map_input_blob_name2shape_)
  {
    const auto &blob_name                  = p_name_shape.first;
    const size_t input_index               = map_input_name2index_.at(blob_name);
    map_blob_name2type_[blob_name] = aclmdlGetInputDataType(model_desc_, input_index);
  }
  for (const auto &p_name_shape : map_output_blob_name2shape_)
  {
    const auto &blob_name                   = p_name_shape.first;
    const size_t output_index               = map_output_name2index_.at(blob_name);
    map_blob_name2type_[blob_name] = aclmdlGetOutputDataType(model_desc_, output_index);
  }
}

std::unique_ptr<BlobsTensor> OmInferCore::AllocBlobsBuffer()
{
  CHECK_STATE_THROW(!map_tensor_type_byte_size_.empty(),
                    "[ACL] Tensor type-byte map not initialized!");
  std::unordered_map<std::string, std::unique_ptr<ITensor>> tensor_map;

  auto CreateAclTensor = [this](const std::string              &blob_name,
                                const std::vector<uint64_t>    &blob_shape,
                                aclDataType                     tensor_type) -> std::unique_ptr<OmTensor> {
    CHECK_STATE_THROW(map_tensor_type_byte_size_.count(tensor_type) > 0,
                      "[ACL] Unsupported aclDataType: %d (blob: %s)", tensor_type,
                      blob_name.c_str());
    size_t elem_byte = map_tensor_type_byte_size_.at(tensor_type);
    CHECK_STATE_THROW(elem_byte > 0, "[ACL] Zero byte size for aclDataType: %d (blob: %s)",
                      tensor_type, blob_name.c_str());

    auto tensor                    = std::make_unique<OmTensor>();
    tensor->name_                  = blob_name;
    tensor->byte_size_per_element_ = elem_byte;
    tensor->current_shape_         = ToSizeShape(blob_shape);
    tensor->default_shape_         = ToSizeShape(blob_shape);
    tensor->tensor_data_type_      = tensor_type;

    size_t      total_byte = tensor->GetTensorByteSize();
    std::string shape_str  = OmInferCore::ShapeToString(blob_shape);
    CHECK_STATE_THROW(total_byte > 0, "[ACL] Zero total byte size for blob: %s (shape: %s)",
                      blob_name.c_str(), shape_str.c_str());
    tensor->self_maintain_buffer_host_ = std::make_unique<unsigned char[]>(total_byte);
    tensor->buffer_on_host_            = tensor->self_maintain_buffer_host_.get();

    LOG_DEBUG("[ACL] Alloc Host buffer: blob=%s, type=%d (byte/elem=%zu), shape=%s, total_byte=%zu",
              blob_name.c_str(), tensor_type, elem_byte, shape_str.c_str(), total_byte);
    return tensor;
  };

  for (const auto &[blob_name, blob_shape] : map_input_blob_name2shape_)
  {
    auto tensor = CreateAclTensor(blob_name, blob_shape, map_blob_name2type_.at(blob_name));
    tensor_map.emplace(blob_name, std::move(tensor));
  }
  for (const auto &[blob_name, blob_shape] : map_output_blob_name2shape_)
  {
    auto tensor = CreateAclTensor(blob_name, blob_shape, map_blob_name2type_.at(blob_name));
    tensor_map.emplace(blob_name, std::move(tensor));
  }

  CHECK_STATE_THROW(!tensor_map.empty(), "[ACL] No tensor buffer allocated (input/output empty)!");
  return std::make_unique<BlobsTensor>(std::move(tensor_map));
}

bool OmInferCore::PreProcess(std::shared_ptr<IPipelinePackage> pipeline_unit)
{
  return true;
}

bool OmInferCore::Inference(std::shared_ptr<IPipelinePackage> buffer)
{
  CHECK_STATE(buffer != nullptr, "[ACL] Inference got null buffer (device %d)", device_id_);
  auto blobs_tensor = buffer->GetInferBuffer();
  CHECK_STATE(blobs_tensor != nullptr, "[ACL] Get blobs tensor failed (device %d)", device_id_);

  aclrtMemcpyKind host2device =
      (run_mode_ == ACL_DEVICE) ? ACL_MEMCPY_HOST_TO_DEVICE : ACL_MEMCPY_HOST_TO_HOST;
  aclrtMemcpyKind device2host =
      (run_mode_ == ACL_DEVICE) ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_HOST_TO_HOST;

  std::vector<const char *> input_blob_names;
  for (const auto &[blob_name, blob_shape] : map_input_blob_name2shape_)
  {
    auto tensor = dynamic_cast<OmTensor *>(blobs_tensor->GetTensor(blob_name));
    CHECK_STATE(tensor != nullptr, "[ACL] Inference got invalid input tensor: %s (device %d)",
                blob_name.c_str(), device_id_);

    size_t tensor_byte_size = tensor->GetTensorByteSize();
    CHECK_STATE(map_input_name2index_.count(blob_name) > 0,
                "[ACL] Unknown input blob name: %s (device %d)", blob_name.c_str(), device_id_);
    const size_t input_index = map_input_name2index_.at(blob_name);
    size_t model_input_size = aclmdlGetInputSizeByIndex(model_desc_, input_index);
    CHECK_STATE(tensor_byte_size == model_input_size,
                "[ACL] Input tensor size mismatch: %s (host: %zu, model: %zu, device %d)",
                blob_name.c_str(), tensor_byte_size, model_input_size, device_id_);

    aclDataBuffer *input_data         = aclmdlGetDatasetBuffer(input_dataset_, input_index);
    void          *input_buffer_device = aclGetDataBufferAddr(input_data);
    CHECK_STATE(input_buffer_device != nullptr,
                "[ACL] Get device buffer for input %s failed (device %d)", blob_name.c_str(),
                device_id_);

    aclError ret = aclrtMemcpy(input_buffer_device, model_input_size, tensor->buffer_on_host_,
                               tensor_byte_size, host2device);
    CHECK_STATE(ret == ACL_SUCCESS, "[ACL] Copy input %s host→device failed (device %d), err: %d",
                blob_name.c_str(), device_id_, ret);

    input_blob_names.push_back(blob_name.c_str());
    LOG_DEBUG("[ACL] Processed input blob: %s (size: %zu bytes, device %d)", blob_name.c_str(),
              tensor_byte_size, device_id_);
  }

  aclError ret = aclmdlExecute(model_id_, input_dataset_, output_dataset_);
  CHECK_STATE(ret == ACL_SUCCESS, "[ACL] Model execute failed (device %d), err: %d", device_id_,
              ret);
  LOG_DEBUG("[ACL] Inference executed with %zu input blobs (device %d)", input_blob_names.size(),
            device_id_);

  std::vector<const char *> output_blob_names;
  for (const auto &[blob_name, blob_shape] : map_output_blob_name2shape_)
  {
    auto tensor = dynamic_cast<OmTensor *>(blobs_tensor->GetTensor(blob_name));
    CHECK_STATE(tensor != nullptr, "[ACL] Inference got invalid output tensor: %s (device %d)",
                blob_name.c_str(), device_id_);

    size_t tensor_byte_size  = tensor->GetTensorByteSize();
    CHECK_STATE(map_output_name2index_.count(blob_name) > 0,
                "[ACL] Unknown output blob name: %s (device %d)", blob_name.c_str(), device_id_);
    const size_t output_index = map_output_name2index_.at(blob_name);
    size_t model_output_size = aclmdlGetOutputSizeByIndex(model_desc_, output_index);
    CHECK_STATE(tensor_byte_size == model_output_size,
                "[ACL] Output tensor size mismatch: %s (host: %zu, model: %zu, device %d)",
                blob_name.c_str(), tensor_byte_size, model_output_size, device_id_);

    aclDataBuffer *output_data          = aclmdlGetDatasetBuffer(output_dataset_, output_index);
    void          *output_buffer_device = aclGetDataBufferAddr(output_data);
    CHECK_STATE(output_buffer_device != nullptr,
                "[ACL] Get device buffer for output %s failed (device %d)", blob_name.c_str(),
                device_id_);

    ret = aclrtMemcpy(tensor->buffer_on_host_, tensor_byte_size, output_buffer_device,
                      model_output_size, device2host);
    CHECK_STATE(ret == ACL_SUCCESS,
                "[ACL] Copy output %s device→host failed (device %d), err: %d", blob_name.c_str(),
                device_id_, ret);

    output_blob_names.push_back(blob_name.c_str());
    LOG_DEBUG("[ACL] Processed output blob: %s (size: %zu bytes, device %d)", blob_name.c_str(),
              tensor_byte_size, device_id_);
  }

  ret = aclrtSynchronizeStream(stream_);
  CHECK_STATE(ret == ACL_SUCCESS, "[ACL] Stream sync failed (device %d), err: %d", device_id_,
              ret);

  LOG_DEBUG("[ACL] Inference success (input: %zu, output: %zu blobs, device %d)",
            input_blob_names.size(), output_blob_names.size(), device_id_);
  return true;
}

bool OmInferCore::PostProcess(std::shared_ptr<IPipelinePackage> buffer)
{
  return true;
}

void OmInferCore::DestroyDataset(aclmdlDataset *&dataset, std::vector<void *> &device_buffers)
{
  if (dataset != nullptr)
  {
    const size_t buffer_num = aclmdlGetDatasetNumBuffers(dataset);
    for (size_t i = 0; i < buffer_num; ++i)
    {
      aclDataBuffer *data_buffer = aclmdlGetDatasetBuffer(dataset, i);
      if (data_buffer != nullptr)
      {
        aclDestroyDataBuffer(data_buffer);
      }
    }
    aclmdlDestroyDataset(dataset);
    dataset = nullptr;
  }
  for (void *ptr : device_buffers)
  {
    if (ptr != nullptr)
    {
      aclrtFree(ptr);
    }
  }
  device_buffers.clear();
}

void OmInferCore::ReleaseAclResources()
{
  DestroyDataset(input_dataset_, input_device_buffers_);
  DestroyDataset(output_dataset_, output_device_buffers_);

  if (model_desc_ != nullptr)
  {
    aclmdlDestroyDesc(model_desc_);
    model_desc_ = nullptr;
  }
  if (model_id_ != 0)
  {
    aclmdlUnload(model_id_);
    model_id_ = 0;
  }
  if (stream_ != nullptr)
  {
    aclrtDestroyStream(stream_);
    stream_ = nullptr;
  }
  if (context_ != nullptr)
  {
    aclrtDestroyContext(context_);
    context_ = nullptr;
  }

  aclrtResetDevice(device_id_);
  if (acl_env_acquired_)
  {
    ReleaseAclEnv();
    acl_env_acquired_ = false;
  }
}

std::shared_ptr<BaseInferCore> CreateOmInferCore(
    const std::string                                             model_path,
    const std::unordered_map<std::string, std::vector<uint64_t>> &input_blobs_shape,
    const std::unordered_map<std::string, std::vector<uint64_t>> &output_blobs_shape,
    const int                                                     device_id)
{
  return std::make_shared<OmInferCore>(model_path, input_blobs_shape, output_blobs_shape,
                                       device_id);
}

} // namespace easy_deploy
