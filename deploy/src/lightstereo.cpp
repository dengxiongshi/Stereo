#include <cstdint>
#include <cstring>
#include <opencv2/highgui/highgui.hpp>
#include "stereo/lightstereo.hpp"

namespace easy_deploy {

namespace {

inline float Fp16ToFp32(uint16_t h)
{
  const uint32_t sign = (h >> 15) & 1u;
  const uint32_t exp  = (h >> 10) & 0x1fu;
  const uint32_t mant = h & 0x3ffu;
  uint32_t       f;
  if (exp == 0)
  {
    if (mant == 0)
    {
      f = sign << 31;
    } else
    {
      uint32_t e = 1;
      uint32_t m = mant;
      while ((m & 0x400u) == 0)
      {
        m <<= 1;
        --e;
      }
      m &= 0x3ffu;
      f = (sign << 31) | ((e + 127 - 15) << 23) | (m << 13);
    }
  } else if (exp == 31)
  {
    f = (sign << 31) | 0x7f800000u | (mant << 13);
  } else
  {
    f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

bool ConvertDispOutputToFloat32(ITensor *out_tensor,
                                int      height,
                                int      width,
                                cv::Mat &disp)
{
  CHECK_STATE(out_tensor != nullptr, "[LightStereo] output tensor is null");
  const void *raw = out_tensor->RawPtr();
  CHECK_STATE(raw != nullptr, "[LightStereo] `PostProcess` Got invalid output disp ptr !!!");

  const size_t elem_bytes = out_tensor->GetElementByteSize();
  const int    n          = height * width;
  disp.create(height, width, CV_32FC1);
  float *dst = disp.ptr<float>();

  if (elem_bytes == 4)
  {
    std::memcpy(dst, raw, static_cast<size_t>(n) * sizeof(float));
  } else if (elem_bytes == 2)
  {
    const auto *src = reinterpret_cast<const uint16_t *>(raw);
    for (int i = 0; i < n; ++i)
    {
      dst[i] = Fp16ToFp32(src[i]);
    }
  } else if (elem_bytes == 1)
  {
    // UINT8: already final quantized values, use directly
    const auto *src = reinterpret_cast<const uint8_t *>(raw);
    for (int i = 0; i < n; ++i)
    {
      dst[i] = static_cast<float>(src[i]);
    }
  } else
  {
    LOG_ERROR("[LightStereo] unsupported output elem bytes: %zu", elem_bytes);
    return false;
  }
  return true;
}

} // namespace

class LightStereo : public BaseStereoMatchingModel {
public:
  LightStereo(const std::shared_ptr<BaseInferCore>    &infer_core,
              const std::shared_ptr<IImageProcessing> &preprocess_block,
              const int                                input_height,
              const int                                input_width,
              const std::vector<std::string>          &input_blobs_name,
              const std::vector<std::string>          &output_blobs_name);

  ~LightStereo() = default;

private:
  bool PreProcess(std::shared_ptr<IPipelinePackage> pipeline_unit) override;

  bool PostProcess(std::shared_ptr<IPipelinePackage> pipeline_unit) override;

private:
  const std::vector<std::string> input_blobs_name_;
  const std::vector<std::string> output_blobs_name_;
  const int                      input_height_;
  const int                      input_width_;

  const std::shared_ptr<BaseInferCore> infer_core_;
  std::shared_ptr<IImageProcessing>    preprocess_block_;
};

LightStereo::LightStereo(const std::shared_ptr<BaseInferCore>    &infer_core,
                         const std::shared_ptr<IImageProcessing> &preprocess_block,
                         const int                                input_height,
                         const int                                input_width,
                         const std::vector<std::string>          &input_blobs_name,
                         const std::vector<std::string>          &output_blobs_name)
    : BaseStereoMatchingModel(infer_core),
      infer_core_(infer_core),
      preprocess_block_(preprocess_block),
      input_height_(input_height),
      input_width_(input_width),
      input_blobs_name_(input_blobs_name),
      output_blobs_name_(output_blobs_name)
{
  // Check if the input arguments and inference_core matches
  auto blobs_tensor = infer_core_->AllocBlobsBuffer();
  if (blobs_tensor->Size() != input_blobs_name_.size() + output_blobs_name_.size())
  {
    LOG_ERROR("[LightStereo] Infer core should has {%d} blobs, but got {%d} blobs!",
              input_blobs_name_.size() + output_blobs_name_.size(), blobs_tensor->Size());
    throw std::runtime_error("[LightStereo] Got invalid input arguments!!");
  }

  for (const std::string &input_blob_name : input_blobs_name)
  {
    blobs_tensor->GetTensor(input_blob_name);
  }

  for (const std::string &output_blob_name : output_blobs_name)
  {
    blobs_tensor->GetTensor(output_blob_name);
  }
}

bool LightStereo::PreProcess(std::shared_ptr<IPipelinePackage> _package)
{
  auto package = std::dynamic_pointer_cast<StereoPipelinePackage>(_package);
  CHECK_STATE(package != nullptr,
              "[LightStereo] PreProcess the `_package` instance does not belong to "
              "`DetectionPipelinePackage`");

  auto blobs_tensor = package->GetInferBuffer();

  const float left_scale  = preprocess_block_->Process(package->left_image_data,
                                                       blobs_tensor->GetTensor(input_blobs_name_[0]),
                                                       input_height_, input_width_);
  const float right_scale = preprocess_block_->Process(
      package->right_image_data, blobs_tensor->GetTensor(input_blobs_name_[1]), input_height_,
      input_width_);

  package->transform_scale = left_scale;
  return true;
}

bool LightStereo::PostProcess(std::shared_ptr<IPipelinePackage> _package)
{
  auto package = std::dynamic_pointer_cast<StereoPipelinePackage>(_package);
  CHECK_STATE(package != nullptr,
              "[LightStereo] PostProcess the `_package` instance does not belong to "
              "`DetectionPipelinePackage`");

  auto p_blob_buffers = package->GetInferBuffer();
  auto *out_tensor    = p_blob_buffers->GetTensor(output_blobs_name_[0]);

  cv::Mat disp;
  CHECK_STATE(ConvertDispOutputToFloat32(out_tensor, input_height_, input_width_, disp),
              "[LightStereo] convert output disp failed");
  cv::imwrite("disp.bmp", disp);
  //  disp /= package->transform_scale;

  // 1. crop
  const int original_height = package->left_image_data->GetImageDataInfo().image_height;
  const int original_width  = package->left_image_data->GetImageDataInfo().image_width;
  const int crop_height     = original_height * package->transform_scale;
  const int crop_width      = original_width * package->transform_scale;
  const int top_pad         = input_height_ - crop_height;
  cv::Mat   crop_disp       = disp(cv::Rect(0, top_pad, crop_width, crop_height));

  // 2. resize to original
  cv::Mat disp_to_original;
  cv::resize(crop_disp, disp_to_original, {original_width, original_height});
  cv::imwrite("disp_to_original.bmp", disp_to_original);
  package->disp = disp_to_original;

  return true;
}

std::shared_ptr<BaseStereoMatchingModel> CreateLightStereoModel(
    const std::shared_ptr<BaseInferCore>    &infer_core,
    const std::shared_ptr<IImageProcessing> &preprocess_block,
    const int                                input_height,
    const int                                input_width,
    const std::vector<std::string>          &input_blobs_name,
    const std::vector<std::string>          &output_blobs_name)
{
  return std::make_shared<LightStereo>(infer_core, preprocess_block, input_height, input_width,
                                       input_blobs_name, output_blobs_name);
}

} // namespace easy_deploy
