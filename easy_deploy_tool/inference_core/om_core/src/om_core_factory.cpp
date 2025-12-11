#include "om_core/om_core.hpp"

namespace easy_deploy {

struct OmInferCoreParams {
  std::string                                            model_path;
  std::unordered_map<std::string, std::vector<uint64_t>> input_blobs_shape;
  std::unordered_map<std::string, std::vector<uint64_t>> output_blobs_shape;
  int                                                    device_id;
};

class OmInferCoreFactory : public BaseInferCoreFactory {
public:
  OmInferCoreFactory(const OmInferCoreParams &params) : params_(params)
  {}

  std::shared_ptr<BaseInferCore> Create() override
  {
    return CreateOmInferCore(params_.model_path, params_.input_blobs_shape,
                              params_.output_blobs_shape, params_.device_id);
  }

private:
  const OmInferCoreParams params_;
};

std::shared_ptr<BaseInferCoreFactory> CreateOmInferCoreFactory(
    const std::string                                             model_path,
    const std::unordered_map<std::string, std::vector<uint64_t>> &input_blobs_shape,
    const std::unordered_map<std::string, std::vector<uint64_t>> &output_blobs_shape,
    const int                                                     device_id)
{
  OmInferCoreParams params;
  params.model_path          = model_path;
  params.input_blobs_shape  = input_blobs_shape;
  params.output_blobs_shape = output_blobs_shape;
  params.device_id        = device_id;

  return std::make_shared<OmInferCoreFactory>(params);
}

} // namespace easy_deploy
