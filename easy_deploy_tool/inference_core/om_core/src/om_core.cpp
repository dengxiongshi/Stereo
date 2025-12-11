#include <unordered_map>
#include <sstream>
#include "om_core/om_core.hpp"
#include "om_blob_buffer.hpp"
#include "acl/acl.h"

using namespace std;

namespace easy_deploy {


    class OmInferCore : public BaseInferCore {
    public:
        ~OmInferCore() override {
            ReleaseAclResources();  // 析构时释放资源
        }

        OmInferCore(const std::string model_path,
                    const std::unordered_map<std::string, std::vector<uint64_t>> &input_blobs_shape,
                    const std::unordered_map<std::string, std::vector<uint64_t>> &output_blobs_shape,
                    const int device_id = 0)
                : model_path_(model_path), device_id_(device_id), map_input_blob_name2shape_(input_blobs_shape),
                  map_output_blob_name2shape_(output_blobs_shape) {
            InitAclResources();
        }

        OmInferCore(const std::string model_path, const int device_id = 0)
                : model_path_(model_path), device_id_(device_id) {
            map_input_blob_name2shape_ = ResolveModelInputInformation();
            map_output_blob_name2shape_ = ResolveModelOutputInformation();
            InitAclResources();  // 初始化ACL核心资源
        }

        std::unique_ptr<BlobsTensor> AllocBlobsBuffer() override;

        InferCoreType GetType() {
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

        std::unordered_map<std::string, std::vector<uint64_t>> ResolveModelInputInformation();

        std::unordered_map<std::string, std::vector<uint64_t>> ResolveModelOutputInformation();
        std::string ShapeToString(const std::vector<uint64_t>& shape);
        // 初始化ACL资源（提取为独立方法，不含预处理参数）
        void InitAclResources();

        // 释放ACL资源
        void ReleaseAclResources();

        void InitTensorTypeByteSize();

    private:
        // ACL核心资源
        int32_t device_id_;
        aclrtContext context_ = nullptr;
        aclrtStream stream_ = nullptr;
        uint32_t model_id_ = 0;
        aclmdlDesc *model_desc_ = nullptr;
        aclrtRunMode run_mode_;

        std::string model_path_;
        std::unordered_map<aclDataType, size_t> map_tensor_type_byte_size_;
        std::unordered_map<std::string, std::vector<uint64_t>> map_input_blob_name2shape_;
        std::unordered_map<std::string, std::vector<uint64_t>> map_output_blob_name2shape_;

        // 输入输出缓冲区（设备端+主机端，外部预处理后的数据直接写入host缓冲区）
        aclmdlDataset *input_dataset_ = nullptr;
        aclmdlDataset *output_dataset_ = nullptr;
        void *input_buffer_device_ = nullptr;
        void *output_buffer_device_ = nullptr;
        size_t input_buffer_size_ = 0;
        size_t output_buffer_size_ = 0;
    };

    std::string OmInferCore::ShapeToString(const std::vector<uint64_t>& shape) {
        std::stringstream ss;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) ss << ",";
            ss << shape[i];
        }
        return ss.str();
    }

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
        ret = aclmdlLoadFromFile(model_path_.c_str(), &model_id_);
        CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] aclmdlLoadFromFile failed, err: %d", ret);

        // 5. 创建模型描述
        model_desc_ = aclmdlCreateDesc();
        ret = aclmdlGetDesc(model_desc_, model_id_);
        CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] aclmdlGetDesc failed, err: %d", ret);

        // 6. 显示输入输出信息
        auto display_blob_info = [](const std::unordered_map<std::string, std::vector<uint64_t>> &blobs) {
            for (const auto &[name, shape]: blobs) {
                std::string shape_str;
                for (auto dim: shape) shape_str += std::to_string(dim) + "\t";
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
        aclDataBuffer *input_data = aclCreateDataBuffer(input_buffer_device_, input_buffer_size_);
        ret = aclmdlAddDatasetBuffer(input_dataset_, input_data);
        CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] Add input dataset buffer failed, err: %d", ret);

        output_buffer_size_ = aclmdlGetOutputSizeByIndex(model_desc_, 0);
        ret = aclrtMalloc(&output_buffer_device_, output_buffer_size_, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] Malloc output device buffer failed, err: %d", ret);

        output_dataset_ = aclmdlCreateDataset();
        aclDataBuffer *output_data = aclCreateDataBuffer(output_buffer_device_, output_buffer_size_);
        ret = aclmdlAddDatasetBuffer(output_dataset_, output_data);
        CHECK_STATE_THROW(ret == ACL_SUCCESS, "[ACL] Add output dataset buffer failed, err: %d", ret);

        BaseInferCore::Init();
        LOG_DEBUG("ACL core initialized successfully");
    }

// 解析模型输入信息
    std::unordered_map<std::string, std::vector<uint64_t>> OmInferCore::ResolveModelInputInformation() {
        std::unordered_map<std::string, std::vector<uint64_t>> input_shape_map;
        size_t input_count = aclmdlGetNumInputs(model_desc_);
        CHECK_STATE_THROW(input_count > 0, "[ACL] No input blobs found");

        for (size_t input_idx = 0; input_idx < input_count; ++input_idx) {
            const char *raw_input_name = aclmdlGetInputNameByIndex(model_desc_, input_idx);
            CHECK_STATE_THROW(raw_input_name != nullptr && raw_input_name[0] != '\0',
                              "[ACL] Failed to get input name for index %zu", input_idx);

            std::string input_name(raw_input_name);
            aclmdlIODims input_dims{};
            aclError ret_acl = aclmdlGetInputDims(model_desc_, input_idx, &input_dims);
            CHECK_STATE_THROW(ret_acl == ACL_SUCCESS, "[ACL] Get input dims failed, err: %d", ret_acl);

            std::vector<uint64_t> input_shape;
            for (size_t dim_idx = 0; dim_idx < input_dims.dimCount; ++dim_idx) {
                int64_t dim_val = input_dims.dims[dim_idx];
                CHECK_STATE_THROW(dim_val > 0, "[ACL] Invalid input dim %d (must be >0)", dim_val);
                input_shape.push_back(static_cast<uint64_t>(dim_val));
            }

            input_shape_map[input_name] = input_shape;
            LOG_DEBUG("[ACL] Resolved input tensor: name=%s, dim_count=%zu, shape=[%s]",
                      input_name.c_str(), input_dims.dimCount,
                      fmt::format("{}", fmt::join(input_shape, ", ")).c_str());
        }

        CHECK_STATE_THROW(!input_shape_map.empty(), "[ACL] Resolved empty input shape map (bug!)");
        return input_shape_map;
    }

// 解析模型输出信息
    std::unordered_map<std::string, std::vector<uint64_t>> OmInferCore::ResolveModelOutputInformation() {
        std::unordered_map<std::string, std::vector<uint64_t>> output_shape_map;
        size_t output_count = aclmdlGetNumOutputs(model_desc_);
        CHECK_STATE_THROW(output_count > 0, "[ACL] No output blobs found");

        for (size_t output_idx = 0; output_idx < output_count; ++output_idx) {
            const char *raw_output_name = aclmdlGetOutputNameByIndex(model_desc_, output_idx);
            CHECK_STATE_THROW(raw_output_name != nullptr && raw_output_name[0] != '\0',
                              "[ACL] Failed to get output name for index %zu", output_idx);
            std::string output_name(raw_output_name);

            aclmdlIODims output_dims{};
            aclError acl_ret = aclmdlGetOutputDims(model_desc_, output_idx, &output_dims);
            CHECK_STATE_THROW(acl_ret == ACL_SUCCESS,
                              "[ACL] aclmdlGetOutputDims failed for output[%zu] (%s), err_code: %d",
                              output_idx, output_name.c_str(), acl_ret);

            CHECK_STATE_THROW(output_dims.dimCount > 0 && output_dims.dimCount <= ACL_MAX_DIM_CNT,
                              "[ACL] Invalid dim count for output[%zu] (%s): dimCount = %zu (max: %d)",
                              output_idx, output_name.c_str(), output_dims.dimCount, ACL_MAX_DIM_CNT);


            std::vector<uint64_t> output_shape;
            for (size_t dim_idx = 0; dim_idx < output_dims.dimCount; ++dim_idx) {
                int64_t dim_val = output_dims.dims[dim_idx];
                CHECK_STATE_THROW(dim_val > 0, "[ACL] Invalid negative dim for output[%zu][dim%zu] (%s): %lld",
                                  output_idx, dim_idx, output_name.c_str(), dim_val);
                output_shape.push_back(static_cast<uint64_t>(dim_val));
            }

            output_shape_map[output_name] = output_shape;
            LOG_DEBUG("[ACL] Resolved output tensor: name=%s, dim_count=%zu, shape=[%s]",
                      output_name.c_str(), output_dims.dimCount,
                      fmt::format("{}", fmt::join(output_shape, ", ")).c_str());
        }

        CHECK_STATE_THROW(!output_shape_map.empty(), "[ACL] Resolved empty output shape map (bug!)");
        return output_shape_map;
    }

    void OmInferCore::InitTensorTypeByteSize() {
        // 映射ACL原生数据类型到字节大小（按需扩展）
        map_tensor_type_byte_size_ = {
                {ACL_FLOAT,   4},        // float → 4字节
                {ACL_INT32,   4},        // int32 → 4字节
                {ACL_UINT8,   1},        // uint8 → 1字节
                {ACL_FLOAT16, 2},      // float16 → 2字节
                {ACL_INT8,    1},         // int8 → 1字节
                {ACL_INT64,   8},        // int64 → 8字节
                {ACL_UINT64,  8}        // uint64 → 8字节
        };
    }

    // 分配输入输出缓冲区（供外部写入预处理后的数据）
    std::unique_ptr<BlobsTensor> OmInferCore::AllocBlobsBuffer() {
        CHECK_STATE_THROW(!map_tensor_type_byte_size_.empty(), "[ACL] Tensor type-byte map not initialized!")
        std::unordered_map<std::string, std::unique_ptr<ITensor>> tensor_map;

        // 分配输入Tensor（外部将预处理后的数据写入此host缓冲区）
        // 提取重复逻辑为Lambda，减少冗余
        auto CreateAclTensor = [this](const std::string& blob_name,
                                      const std::vector<uint64_t>& blob_shape,
                                      aclDataType tensor_type) -> std::unique_ptr<OmTensor> {
            // 检查类型是否支持
            CHECK_STATE_THROW(map_tensor_type_byte_size_.count(tensor_type) > 0,
                              "[ACL] Unsupported aclDataType: %d (blob: %s)",
                              tensor_type, blob_name.c_str());
            // 检查类型字节数有效（排除未定义/0字节）
            size_t elem_byte = map_tensor_type_byte_size_.at(tensor_type);
            CHECK_STATE_THROW(elem_byte > 0, "[ACL] Zero byte size for aclDataType: %d (blob: %s)",
                              tensor_type, blob_name.c_str());

            auto tensor = std::make_unique<OmTensor>();
            tensor->name_ = blob_name;
            tensor->byte_size_per_element_ = elem_byte;
            tensor->current_shape_ = blob_shape;
            tensor->default_shape_ = blob_shape;
            tensor->tensor_data_type_ = tensor_type;

            size_t total_byte = tensor->GetTensorByteSize();
            std::string shape_str = OmInferCore::ShapeToString(blob_shape);
            CHECK_STATE_THROW(total_byte > 0, "[ACL] Zero total byte size for blob: %s (shape: %s)",
                              blob_name.c_str(), shape_str.c_str());
            tensor->self_maintain_buffer_host_ = std::make_unique<unsigned char[]>(total_byte);
            tensor->buffer_on_host_ = tensor->self_maintain_buffer_host_.get();


            LOG_DEBUG("[ACL] Alloc Host buffer: blob=%s, type=%d (byte/elem=%zu), shape=%s, total_byte=%zu",
                      blob_name.c_str(), tensor_type, elem_byte, shape_str.c_str(), total_byte);

            return tensor;
        };

        // 1. 分配输入张量（默认类型：ACL_FLOAT，可根据模型动态调整）
        for (const auto& [blob_name, blob_shape] : map_input_blob_name2shape_) {
            auto tensor = CreateAclTensor(blob_name, blob_shape, ACL_FLOAT);
            tensor_map.emplace(blob_name, std::move(tensor));
        }

        // 2. 分配输出张量（默认类型：ACL_FLOAT，可根据模型动态调整）
        for (const auto& [blob_name, blob_shape] : map_output_blob_name2shape_) {
            auto tensor = CreateAclTensor(blob_name, blob_shape, ACL_FLOAT);
            tensor_map.emplace(blob_name, std::move(tensor));
        }

        CHECK_STATE_THROW(!tensor_map.empty(), "[ACL] No tensor buffer allocated (input/output empty)!");

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
        std::vector<const char *> input_blob_names;  // 输入blob名称列表（与模型输入顺序对应）
        for (const auto &[blob_name, blob_shape]: map_input_blob_name2shape_) {
            // 获取当前输入tensor
            auto tensor = dynamic_cast<OmTensor *>(blobs_tensor->GetTensor(blob_name));
            CHECK_STATE(tensor != nullptr, "[ACL] Inference got invalid input tensor: %s (device %d)",
                        blob_name.c_str(), device_id_);

            // 校验输入tensor大小与模型要求一致
            size_t tensor_byte_size = tensor->GetTensorByteSize();
            int input_index = std::stoi(blob_name.substr(6));  // 从"input_N"提取索引N
            size_t model_input_size = aclmdlGetInputSizeByIndex(model_desc_, input_index);
            CHECK_STATE(tensor_byte_size == model_input_size,
                        "[ACL] Input tensor size mismatch: %s (host: %zu, model: %zu, device %d)",
                        blob_name.c_str(), tensor_byte_size, model_input_size, device_id_);

            // 获取当前输入对应的设备端缓冲区（从输入数据集提取）
            aclDataBuffer *input_data = aclmdlGetDatasetBuffer(input_dataset_, input_index);
            void *input_buffer_device = aclGetDataBufferAddr(input_data);
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
        std::vector<const char *> output_blob_names;  // 输出blob名称列表（与模型输出顺序对应）
        for (const auto &[blob_name, blob_shape]: map_output_blob_name2shape_) {
            // 获取当前输出tensor
            auto tensor = dynamic_cast<OmTensor *>(blobs_tensor->GetTensor(blob_name));
            CHECK_STATE(tensor != nullptr, "[ACL] Inference got invalid output tensor: %s (device %d)",
                        blob_name.c_str(), device_id_);

            // 校验输出tensor大小与模型要求一致
            size_t tensor_byte_size = tensor->GetTensorByteSize();
            int output_index = std::stoi(blob_name.substr(7));  // 从"output_N"提取索引N
            size_t model_output_size = aclmdlGetOutputSizeByIndex(model_desc_, output_index);
            CHECK_STATE(tensor_byte_size == model_output_size,
                        "[ACL] Output tensor size mismatch: %s (host: %zu, model: %zu, device %d)",
                        blob_name.c_str(), tensor_byte_size, model_output_size, device_id_);

            // 获取当前输出对应的设备端缓冲区（从输出数据集提取）
            aclDataBuffer *output_data = aclmdlGetDatasetBuffer(output_dataset_, output_index);
            void *output_buffer_device = aclGetDataBufferAddr(output_data);
            CHECK_STATE(output_buffer_device != nullptr,
                        "[ACL] Get device buffer for output %s failed (device %d)", blob_name.c_str(), device_id_);

            uint32_t len = aclGetDataBufferSizeV2(output_data);

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
            const std::string model_path,
            const std::unordered_map<std::string, std::vector<uint64_t>> &input_blobs_shape,
            const std::unordered_map<std::string, std::vector<uint64_t>> &output_blobs_shape,
            const int device_id) {
        return std::make_shared<OmInferCore>(model_path, input_blobs_shape, output_blobs_shape, device_id);
    }

}
