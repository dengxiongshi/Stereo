/**
 * @Copyright (C) ShenZhen ShenZhi WeiLai Co., Ltd. 2017-2025. All rights reserved.
 * @Date 2025/12/27 16:31.
 * @author 邓熊狮
 * @FlieName test
 * @description:
 **/

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <dirent.h>
#include "acl/acl.h"

#include "deploy_core/base_stereo.hpp"
#include "deploy_core/wrapper.hpp"

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;


class SampleStereo {
    public:
        SampleStereo(int32_t device, const std::string model_path, int32_t modelWidth, int32_t modelHeight);
        ~SampleStereo();
        Result InitResource();
        Result ProcessInput(const std::string leftPath, const std::string rightPath);
        Result Inference();
        Result GetResult();

    private:
        void ReleaseResource();
        int32_t deviceId_;
        aclrtContext context_;
        aclrtStream stream_;

        uint32_t modelId_;
        const char* modelPath_;
        int32_t modelWidth_;
        int32_t modelHeight_;
        aclmdlDesc *modelDesc_;
        aclmdlDataset *inputDataset_;
        aclmdlDataset *outputDataset_;
        void* inputBuffer_;
        void *outputBuffer_;
        size_t inputBufferSize_;
        float* imageBytes;
        std::string imagePath_1;
        std::string imagePath_2;
        cv::Mat leftImage;
        cv::Mat rightImage;
        aclrtRunMode runMode_;
};

SampleStereo::SampleStereo(int32_t device, const std::string model_path, int32_t modelWidth, int32_t modelHeight) :
deviceId_(device), context_(nullptr), stream_(nullptr), modelId_(0),
modelPath_(modelPath), modelWidth_(modelWidth), modelHeight_(modelHeight),
modelDesc_(nullptr), inputDataset_(nullptr), outputDataset_(nullptr)
{
}

SampleResnetQuickStart::~SampleResnetQuickStart() {
    ReleaseResource();
}

Result SampleStereo::InitResource() {
    // init acl resource
    const char *aclConfigPath = "";
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclInit failed, errorCode is %d", ret);
        return FAILED;
    }
    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtSetDevice failed, errorCode is %d", ret);
        return FAILED;
    }
    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtCreateContext failed, errorCode is %d", ret);
        return FAILED;
    }
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtCreateStream failed, errorCode is %d", ret);
        return FAILED;
    }
    
    // load model file
    ret = aclmdlLoadFromFile(modelPath_, &modelId_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclmdLoadFromFile failed, errorCode is %d", ret);
        return FAILED;
    }
    // create description of model
    modelDesc_ = aclmdlCreateDesc();
    ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclmdlGetDesc failed, errorCode is %d", ret);
        return FAILED;
    }
    ret = aclrtGetRunMode(&runMode_);
    if (ret == FAILED) {
        ERROR_LOG("get runMode failed, errorCode is %d", ret);
        return FAILED;
    }
    // create data set of input
    inputDataset_ = aclmdlCreateDataset();
    size_t inputIndex = 0;
    inputBufferSize_ = aclmdlGetInputSizeByIndex(modelDesc_, inputIndex);
    aclrtMalloc(&inputBuffer_, inputBufferSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    aclDataBuffer *inputData = aclCreateDataBuffer(inputBuffer_, inputBufferSize_);
    ret = aclmdlAddDatasetBuffer(inputDataset_, inputData);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclmdlAddDatasetBuffer failed, errorCode is %d", ret);
        return FAILED;
    }

    // create data set of output
    outputDataset_ = aclmdlCreateDataset();
    size_t outputIndex = 0;
    size_t modelOutputSize = aclmdlGetOutputSizeByIndex(modelDesc_, outputIndex);
    aclrtMalloc(&outputBuffer_, modelOutputSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer_, modelOutputSize);
    ret = aclmdlAddDatasetBuffer(outputDataset_, outputData);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclmdlAddDatasetBuffer failed, errorCode is %d", ret);
        return FAILED;
    }
    
    return SUCCESS;
}

Result SampleStereo::ProcessInput(const std::string leftPath, const std::string rightPath) {
    leftImage = cv::imread(leftPath);
    rightImage = cv::imread(rightPath);
    auto package = std::make_shared<StereoPipelinePackage>();
    package->left_image_data  = std::make_shared<PipelineCvImageWrapper>(left_image);
    package->right_image_data = std::make_shared<PipelineCvImageWrapper>(right_image);

    auto preprocess_block = CreateCpuImageProcessingResizePad(
        ImageProcessingPadMode::TOP_RIGHT,
        ImageProcessingPadValue::EDGE,
        true, true, {0, 0, 0}, {1, 1, 1});

    
}