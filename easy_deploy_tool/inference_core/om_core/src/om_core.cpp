#include <unordered_map>
#include "om_core/om_core.hpp"
#include "om_blob_buffer.hpp"
#include "acl/acl.h"
#include "acllite_dvpp_lite/ImageProc.h"
#include "acllite_om_execute/ModelProc.h"

using namespace std;
using namespace acllite;

namespace easy_deploy {

// static std::unordered_map<OmInputTensorType, om_tensor_type> map_type_my2om{
//     {OmInputTensorType::OM_UINT8, OM_TENSOR_UINT8},
//     {OmInputTensorType::OM_INT8, OM_TENSOR_INT8},
//     {OmInputTensorType::OM_FLOAT16, OM_TENSOR_FLOAT16},
//     {OmInputTensorType::OM_FLOAT32, OM_TENSOR_FLOAT32},
//     {OmInputTensorType::OM_UINT32, OM_TENSOR_UINT32},
//     {OmInputTensorType::OM_INT32, OM_TENSOR_INT32},
//     {OmInputTensorType::OM_INT64, OM_TENSOR_INT64},
// };

static const std::unordered_map<OMTensorElementDataType, size_t> map_tensor_type_byte_size_{
    {OMTensorElementDataType::OM_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4},
    {OMTensorElementDataType::OM_TENSOR_ELEMENT_DATA_TYPE_INT32, 4},
    {OMTensorElementDataType::OM_TENSOR_ELEMENT_DATA_TYPE_UINT32, 4},
    {OMTensorElementDataType::OM_TENSOR_ELEMENT_DATA_TYPE_INT64, 8},
    {OMTensorElementDataType::OM_TENSOR_ELEMENT_DATA_TYPE_UINT64, 8}};

// static std::unordered_map<om_tensor_type, int> map_om_type2size_{
//     {OM_TENSOR_INT8, 1},    {OM_TENSOR_UINT8, 1}, {OM_TENSOR_FLOAT16, 4},
//     {OM_TENSOR_FLOAT32, 4}, {OM_TENSOR_INT32, 4}, {OM_TENSOR_UINT32, 4},
//     {OM_TENSOR_INT64, 8}};

// static std::unordered_map<om_tensor_type, om_tensor_type> map_om_type2type{
//     {OM_TENSOR_INT8, OM_TENSOR_UINT8},      {OM_TENSOR_UINT8, OM_TENSOR_UINT8},
//     {OM_TENSOR_FLOAT16, OM_TENSOR_FLOAT32}, {OM_TENSOR_FLOAT32, OM_TENSOR_FLOAT32},
//     {OM_TENSOR_INT32, OM_TENSOR_INT32},     {OM_TENSOR_UINT32, OM_TENSOR_UINT32},
//     {OM_TENSOR_INT64, OM_TENSOR_INT64}};

class OmInferCore : public BaseInferCore {
public:
  ~OmInferCore() override {
    ReleaseAclResources();  // 析构时释放资源
  }
  OmInferCore(const std::string                                             model_path,
              const std::unordered_map<std::string, std::vector<uint64_t>> &input_blobs_shape,
              const std::unordered_map<std::string, std::vector<uint64_t>> &output_blobs_shape,
              const int                                                    device_id = 0)
    : model_path_(model_path), device_id_(device_id), map_input_blob_name2shape_(input_blobs_shape), map_output_blob_name2shape_(output_blobs_shape) {
    InitAclResources();
  }

  OmInferCore(const std::string model_path, const int decice_id = 0)
    : model_path_(model_path), device_id_(device_id) {
    map_input_blob_name2shape_ = ResolveModelInputInformation();
    map_output_blob_name2shape_ = ResolveModelOutputInformation();
    InitAclResources();  // 初始化ACL核心资源
  }

  std::unique_ptr<BlobsTensor> AllocBlobsBuffer() override;

  InferCoreType GetType()
  {
    return InferCoreType::OM;
  }

    std::string GetName() {
      return "om_core";
    }

private:
  bool PreProcess(std::shared_ptr<IPipelinePackage> buffer) override;

  bool Inference(std::shared_ptr<IPipelinePackage> buffer) override;

  bool PostProcess(std::shared_ptr<IPipelinePackage> buffer) override;

private:

  size_t ReadModelFromFile(const std::string &model_path, ModelProc modelProc);
  
  std::unordered_map<std::string, std::vector<uint64_t>> ResolveModelInputInformation();

  std::unordered_map<std::string, std::vector<uint64_t>> ResolveModelOutputInformation();
//   void ResolveModelInformation(const std::unordered_map<std::string, OmInputTensorType> &map_blob_type);
  // 初始化ACL资源（提取为独立方法，不含预处理参数）
  void InitAclResources();

  // 释放ACL资源
  void ReleaseAclResources();

private:
  // ACL核心资源
  int32_t device_id_;
  aclrtContext context_ = nullptr;
  aclrtStream stream_ = nullptr;
  uint32_t model_id_ = 0;
  aclmdlDesc* model_desc_ = nullptr;
  aclrtRunMode run_mode_;

  std::string model_path_;

  std::unordered_map<std::string, std::vector<uint64_t>> map_input_blob_name2shape_;
  std::unordered_map<std::string, std::vector<uint64_t>> map_output_blob_name2shape_;

  // 输入输出缓冲区（设备端+主机端，外部预处理后的数据直接写入host缓冲区）
  aclmdlDataset* input_dataset_ = nullptr;
  aclmdlDataset* output_dataset_ = nullptr;
  void* input_buffer_device_ = nullptr;
  void* output_buffer_device_ = nullptr;
  size_t input_buffer_size_ = 0;
  size_t output_buffer_size_ = 0;
};

void OmInferCore::InitAclResources() {
  LOG_DEBUG("Initializing ACL core with model {%s} ...", model_path_.c_str());
  // 1. 初始化ACL环境
  aclError ret = aclInit("");
  CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] aclInit failed, err: %d", ret);

  // 2. 设置设备与创建上下文
  ret = aclrtSetDevice(device_id_);
  CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] aclrtSetDevice failed, err: %d", ret);

  ret = aclrtCreateContext(&context_, device_id_);
  CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] aclrtCreateContext failed, err: %d", ret);

  ret = aclrtCreateStream(&stream_);
  CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] aclrtCreateStream failed, err: %d", ret);

  // 3. 获取运行模式
  ret = aclrtGetRunMode(&run_mode_);
  CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] aclrtGetRunMode failed, err: %d", ret);

  // 4. 加载模型
  ret = aclmdlLoadFromFile(acl_path_.c_str(), &model_id_);
  CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] aclmdlLoadFromFile failed, err: %d", ret);

  // 5. 创建模型描述
  model_desc_ = aclmdlCreateDesc();
  ret = aclmdlGetDesc(model_desc_, model_id_);
  CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] aclmdlGetDesc failed, err: %d", ret);

  // 6. 显示输入输出信息
  auto display_blob_info = [](const std::unordered_map<std::string, std::vector<uint64_t>>& blobs) {
    for (const auto& [name, shape] : blobs) {
      std::string shape_str;
      for (auto dim : shape) shape_str += std::to_string(dim) + "\t";
      LOG_DEBUG("blob name: %s, shape: %s", name.c_str(), shape_str.c_str());
    }
  };
  display_blob_info(map_input_blob_name2shape_);
  display_blob_info(map_output_blob_name2shape_);

  // 7. 初始化输入输出数据集（设备端缓冲区）
  input_buffer_size_ = aclmdlGetInputSizeByIndex(model_desc_, 0);
  ret = aclrtMalloc(&input_buffer_device_, input_buffer_size_, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] Malloc input device buffer failed, err: %d", ret);

  input_dataset_ = aclmdlCreateDataset();
  aclDataBuffer* input_data = aclCreateDataBuffer(input_buffer_device_, input_buffer_size_);
  ret = aclmdlAddDatasetBuffer(input_dataset_, input_data);
  CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] Add input dataset buffer failed, err: %d", ret);

  output_buffer_size_ = aclmdlGetOutputSizeByIndex(model_desc_, 0);
  ret = aclrtMalloc(&output_buffer_device_, output_buffer_size_, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] Malloc output device buffer failed, err: %d", ret);

  output_dataset_ = aclmdlCreateDataset();
  aclDataBuffer* output_data = aclCreateDataBuffer(output_buffer_device_, output_buffer_size_);
  ret = aclmdlAddDatasetBuffer(output_dataset_, output_data);
  CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] Add output dataset buffer failed, err: %d", ret);

  BaseInferCore::Init();
  LOG_DEBUG("ACL core initialized successfully");
}

// 解析模型输入信息
std::unordered_map<std::string, std::vector<uint64_t>> OmInferCore::ResolveModelInputInformation() {
  std::unordered_map<std::string, std::vector<uint64_t>> ret;
  int input_count = aclmdlGetNumInputs(model_desc_);
  CHECK_STATE_THROW(input_count > 0, "[ACL] No input blobs found");

  for (int i = 0; i < input_count; ++i) {
    std::string blob_name = "input_" + std::to_string(i);
    int dim_count = aclmdlGetInputDimCount(model_desc_, i);
    aclmdlDim_t dims[dim_count];
    aclError ret_acl = aclmdlGetInputDims(model_desc_, i, dims, dim_count);
    CHECK_STATE_THROW(ret_acl == ACL_SUCCESS, "[ACL] Get input dims failed, err: %d", ret_acl);

    std::vector<uint64_t> blob_shape;
    for (int d = 0; d < dim_count; ++d) {
      CHECK_STATE_THROW(dims[d] > 0, "[ACL] Invalid input dim %d (must be >0)", dims[d]);
      blob_shape.push_back(static_cast<uint64_t>(dims[d]));
    }

    ret[blob_name] = blob_shape;
    LOG_DEBUG("Resolved input blob: %s, shape: %s", blob_name.c_str(), 
              fmt::format("{}", fmt::join(blob_shape, ", ")).c_str());
  }

  return ret;
}

// 解析模型输出信息
std::unordered_map<std::string, std::vector<uint64_t>> OmInferCore::ResolveModelOutputInformation() {
  std::unordered_map<std::string, std::vector<uint64_t>> ret;
  int output_count = aclmdlGetNumOutputs(model_desc_);
  CHECK_STATE_THROW(output_count > 0, "[ACL] No output blobs found");

  for (int i = 0; i < output_count; ++i) {
    std::string blob_name = "output_" + std::to_string(i);
    int dim_count = aclmdlGetOutputDimCount(model_desc_, i);
    aclmdlDim_t dims[dim_count];
    aclError ret_acl = aclmdlGetOutputDims(model_desc_, i, dims, dim_count);
    CHECK_STATE_THROW(ret_acl == ACL_SUCCESS, "[ACL] Get output dims failed, err: %d", ret_acl);

    std::vector<uint64_t> blob_shape;
    for (int d = 0; d < dim_count; ++d) {
      CHECK_STATE_THROW(dims[d] > 0, "[ACL] Invalid output dim %d (must be >0)", dims[d]);
      blob_shape.push_back(static_cast<uint64_t>(dims[d]));
    }

    ret[blob_name] = blob_shape;
    LOG_DEBUG("Resolved output blob: %s, shape: %s", blob_name.c_str(), 
              fmt::format("{}", fmt::join(blob_shape, ", ")).c_str());
  }

  return ret;
}

// 分配输入输出缓冲区（供外部写入预处理后的数据）
std::unique_ptr<BlobsTensor> OmInferCore::AllocBlobsBuffer() {
  std::unordered_map<std::string, std::unique_ptr<ITensor>> tensor_map;

  // 分配输入Tensor（外部将预处理后的数据写入此host缓冲区）
  for (const auto& [blob_name, blob_shape] : map_input_blob_name2shape_) {
    auto tensor = std::make_unique<OmTensor>();
    ONNXTensorElementDataType tensor_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    CHECK_STATE_THROW(map_tensor_type_byte_size_.count(tensor_type), 
                      "[ACL] Unsupported input tensor type");

    tensor->name_ = blob_name;
    tensor->byte_size_per_element_ = map_tensor_type_byte_size_.at(tensor_type);
    tensor->current_shape_ = blob_shape;
    tensor->default_shape_ = blob_shape;
    tensor->self_maintain_buffer_host_ = std::make_unique<u_char[]>(tensor->GetTensorByteSize());
    tensor->buffer_on_host_ = tensor->self_maintain_buffer_host_.get();
    tensor->tensor_data_type_ = tensor_type;

    tensor_map.emplace(blob_name, std::move(tensor));
  }

  // 分配输出Tensor（推理结果写入此host缓冲区）
  for (const auto& [blob_name, blob_shape] : map_output_blob_name2shape_) {
    auto tensor = std::make_unique<OmTensor>();
    ONNXTensorElementDataType tensor_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    CHECK_STATE_THROW(map_tensor_type_byte_size_.count(tensor_type), 
                      "[ACL] Unsupported output tensor type");

    tensor->name_ = blob_name;
    tensor->byte_size_per_element_ = map_tensor_type_byte_size_.at(tensor_type);
    tensor->current_shape_ = blob_shape;
    tensor->default_shape_ = blob_shape;
    tensor->self_maintain_buffer_host_ = std::make_unique<u_char[]>(tensor->GetTensorByteSize());
    tensor->buffer_on_host_ = tensor->self_maintain_buffer_host_.get();
    tensor->tensor_data_type_ = tensor_type;

    tensor_map.emplace(blob_name, std::move(tensor));
  }

  return std::make_unique<BlobsTensor>(std::move(tensor_map));
}

bool OmInferCore::PreProcess(std::shared_ptr<IPipelinePackage> pipeline_unit) {
  return true;
}

bool OmInferCore::Inference(std::shared_ptr<IPipelinePackage> buffer) {
  // 校验输入缓冲区有效性
  CHECK_STATE(buffer != nullptr, "[ACL] Inference got null buffer (device %d)", device_id_);
  auto blobs_tensor = buffer->GetInferBuffer();
  CHECK_STATE(blobs_tensor != nullptr, "[ACL] Get blobs tensor failed (device %d)", device_id_);

  // 确定内存拷贝方向（设备模式/主机模式）
  aclrtMemcpyKind host2device = (run_mode_ == ACL_DEVICE) ? ACL_MEMCPY_HOST_TO_DEVICE : ACL_MEMCPY_HOST_TO_HOST;
  aclrtMemcpyKind device2host = (run_mode_ == ACL_DEVICE) ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_HOST_TO_HOST;

  // 处理所有输入blob：从host拷贝到device
  std::vector<const char*> input_blob_names;  // 输入blob名称列表（与模型输入顺序对应）
  for (const auto& [blob_name, blob_shape] : map_input_blob_name2shape_) {
    // 获取当前输入tensor
    auto tensor = dynamic_cast<OmTensor*>(blobs_tensor->GetTensor(blob_name));
    CHECK_STATE(tensor != nullptr, "[ACL] Inference got invalid input tensor: %s (device %d)", blob_name.c_str(), device_id_);

    // 校验输入tensor大小与模型要求一致
    size_t tensor_byte_size = tensor->GetTensorByteSize();
    int input_index = std::stoi(blob_name.substr(6));  // 从"input_N"提取索引N
    size_t model_input_size = aclmdlGetInputSizeByIndex(model_desc_, input_index);
    CHECK_STATE(tensor_byte_size == model_input_size, 
                "[ACL] Input tensor size mismatch: %s (host: %zu, model: %zu, device %d)",
                blob_name.c_str(), tensor_byte_size, model_input_size, device_id_);

    // 获取当前输入对应的设备端缓冲区（从输入数据集提取）
    aclDataBuffer* input_data = aclmdlGetDatasetBuffer(input_dataset_, input_index);
    void* input_buffer_device = aclGetDataBufferAddr(input_data);
    CHECK_STATE(input_buffer_device != nullptr, 
                "[ACL] Get device buffer for input %s failed (device %d)", blob_name.c_str(), device_id_);

    // 拷贝数据：host → device
    aclError ret = aclrtMemcpy(input_buffer_device, model_input_size,
                               tensor->buffer_on_host_, tensor_byte_size,
                               host2device);
    CHECK_STATE(ret == ACL_SUCCESS, 
                "[ACL] Copy input %s host→device failed (device %d), err: %d", 
                blob_name.c_str(), device_id_, ret);

    input_blob_names.push_back(blob_name.c_str());
    LOG_DEBUG("[ACL] Processed input blob: %s (size: %zu bytes, device %d)", 
              blob_name.c_str(), tensor_byte_size, device_id_);
  }

  // 执行模型推理
  aclError ret = aclmdlExecute(model_id_, input_dataset_, output_dataset_);
  CHECK_STATE(ret == ACL_SUCCESS, 
              "[ACL] Model execute failed (device %d), err: %d", device_id_, ret);
  LOG_DEBUG("[ACL] Inference executed with %zu input blobs (device %d)", 
            input_blob_names.size(), device_id_);

  // 处理所有输出blob：从device拷贝到host
  std::vector<const char*> output_blob_names;  // 输出blob名称列表（与模型输出顺序对应）
  for (const auto& [blob_name, blob_shape] : map_output_blob_name2shape_) {
    // 获取当前输出tensor
    auto tensor = dynamic_cast<OrtTensor*>(blobs_tensor->GetTensor(blob_name));
    CHECK_STATE(tensor != nullptr, "[ACL] Inference got invalid output tensor: %s (device %d)", blob_name.c_str(), device_id_);

    // 校验输出tensor大小与模型要求一致
    size_t tensor_byte_size = tensor->GetTensorByteSize();
    int output_index = std::stoi(blob_name.substr(7));  // 从"output_N"提取索引N
    size_t model_output_size = aclmdlGetOutputSizeByIndex(model_desc_, output_index);
    CHECK_STATE(tensor_byte_size == model_output_size, 
                "[ACL] Output tensor size mismatch: %s (host: %zu, model: %zu, device %d)",
                blob_name.c_str(), tensor_byte_size, model_output_size, device_id_);

    // 获取当前输出对应的设备端缓冲区（从输出数据集提取）
    aclDataBuffer* output_data = aclmdlGetDatasetBuffer(output_dataset_, output_index);
    void* output_buffer_device = aclGetDataBufferAddr(output_data);
    CHECK_STATE(output_buffer_device != nullptr, 
                "[ACL] Get device buffer for output %s failed (device %d)", blob_name.c_str(), device_id_);

    // 拷贝数据：device → host
    ret = aclrtMemcpy(tensor->buffer_on_host_, tensor_byte_size,
                      output_buffer_device, model_output_size,
                      device2host);
    CHECK_STATE(ret == ACL_SUCCESS, 
                "[ACL] Copy output %s device→host failed (device %d), err: %d", 
                blob_name.c_str(), device_id_, ret);

    output_blob_names.push_back(blob_name.c_str());
    LOG_DEBUG("[ACL] Processed output blob: %s (size: %zu bytes, device %d)", 
              blob_name.c_str(), tensor_byte_size, device_id_);
  }

  // 同步流确保所有数据拷贝完成
  ret = aclrtSynchronizeStream(stream_);
  CHECK_STATE(ret == ACL_SUCCESS, 
              "[ACL] Stream sync failed (device %d), err: %d", device_id_, ret);

  LOG_DEBUG("[ACL] Inference success (input: %zu, output: %zu blobs, device %d)",
            input_blob_names.size(), output_blob_names.size(), device_id_);
  return true;
}

// 后处理（解析推理结果，保持原有逻辑）
bool OmInferCore::PostProcess(std::shared_ptr<IPipelinePackage> buffer) {
  return true;
}

// 释放ACL资源
void OmInferCore::ReleaseAclResources() {
  if (input_buffer_device_ != nullptr) {
    aclrtFree(input_buffer_device_);
    input_buffer_device_ = nullptr;
  }
  if (output_buffer_device_ != nullptr) {
    aclrtFree(output_buffer_device_);
    output_buffer_device_ = nullptr;
  }

  if (input_dataset_ != nullptr) {
    aclmdlDestroyDataset(input_dataset_);
    input_dataset_ = nullptr;
  }
  if (output_dataset_ != nullptr) {
    aclmdlDestroyDataset(output_dataset_);
    output_dataset_ = nullptr;
  }

  if (model_desc_ != nullptr) {
    aclmdlDestroyDesc(model_desc_);
    model_desc_ = nullptr;
  }
  if (model_id_ != 0) {
    aclmdlUnload(model_id_);
    model_id_ = 0;
  }

  if (stream_ != nullptr) {
    aclrtDestroyStream(stream_);
    stream_ = nullptr;
  }
  if (context_ != nullptr) {
    aclrtDestroyContext(context_);
    context_ = nullptr;
  }

  aclrtResetDevice(device_id_);
  aclFinalize();
}

std::shared_ptr<BaseInferCore> CreateOmInferCore(
    const std::string                                             model_path,
    const std::unordered_map<std::string, std::vector<uint64_t>> &input_blobs_shape,
    const std::unordered_map<std::string, std::vector<uint64_t>> &output_blobs_shape,
    const int                                                    device_id) {
  return std::make_shared<OmInferCore>(model_path, input_blobs_shape, output_blobs_shape, device_id);
}

}
