/**
 * @Copyright (C) ShenZhen ShenZhi WeiLai Co., Ltd. 2017-2025. All rights reserved.
 * @Date 2025/10/31 17:35.
 * @author 邓熊狮
 * @FlieName om_core
 * @description:
 **/

#pragma once

#include "deploy_core/base_infer_core.hpp"

namespace easy_deploy {

std::shared_ptr<BaseInferCore> CreateOmInferCore(
    const std::string                                             model_path,
    const std::unordered_map<std::string, std::vector<uint64_t>> &input_blobs_shape  = {},
    const std::unordered_map<std::string, std::vector<uint64_t>> &output_blobs_shape = {},
    const int                                                    device_id           = 0);

std::shared_ptr<BaseInferCoreFactory> CreateOmInferCoreFactory(
    const std::string                                             model_path,
    const std::unordered_map<std::string, std::vector<uint64_t>> &input_blobs_shape  = {},
    const std::unordered_map<std::string, std::vector<uint64_t>> &output_blobs_shape = {},
    const int                                                    device_id           = 0);

} // namespace easy_deploy
