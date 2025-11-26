/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 * Description:
 */
#ifndef FLOW_FUNC_META_FLOW_FUNC_H
#define FLOW_FUNC_META_FLOW_FUNC_H

#include <vector>
#include <memory>
#include <functional>
#include "flow_func_defines.h"
#include "flow_msg.h"
#include "meta_context.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY MetaFlowFunc {
public:
    MetaFlowFunc() = default;

    virtual ~MetaFlowFunc() = default;

    /**
     * set context.
     * it will be call before Init.
     * @param metaContext context.
     */
    void SetContext(MetaContext *metaContext)
    {
        context_ = metaContext;
    }

    /**
     * @brief Flow func init.
     * only call once.
     * @return 0:success, other: failed.
     */
    virtual int32_t Init() = 0;

    /**
     * @brief flow func proc func.
     * @param inputMsgs input msgs
     * @return FLOW_FUNC_SUCCESS:success, other: failed.
     * If an unrecoverable exception occurs, return error.
     * others can call SetRetCode to out tensor, than return success.
     * if return success, schedule will be break.
     */
    virtual int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &inputMsgs) = 0;

protected:
    MetaContext *context_ = nullptr;
};

using FLOW_FUNC_CREATOR_FUNC = std::function<std::shared_ptr<MetaFlowFunc>(void)>;

/**
 * @brief register flow func creator.
 * @param flowFuncName cannot be null and must end with '\0'
 * @param func flow func creator
 * @return if register success, true:success.
 */
FLOW_FUNC_VISIBILITY bool RegisterFlowFunc(const char *flowFuncName, const FLOW_FUNC_CREATOR_FUNC &func) noexcept;

#define REGISTER_FLOW_FUNC_IMPL(name, ctr, clazz)                   \
    std::shared_ptr<FlowFunc::MetaFlowFunc> Creator_##clazz##ctr##_FlowFunc() \
    {                                                               \
        return std::make_shared<clazz>();                           \
    }                                                               \
    bool g_##clazz##ctr##_FlowFuncCreator __attribute__((unused)) = \
        FlowFunc::RegisterFlowFunc(name, Creator_##clazz##ctr##_FlowFunc)

#define REGISTER_FLOW_FUNC_INNER(name, ctr, clazz) REGISTER_FLOW_FUNC_IMPL(name, ctr, clazz)
#define REGISTER_FLOW_FUNC(name, clazz) REGISTER_FLOW_FUNC_INNER(name, __COUNTER__, clazz)
}  // namespace FlowFunc
#endif // FLOW_FUNC_META_FLOW_FUNC_H
