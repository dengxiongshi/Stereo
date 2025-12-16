/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 * Description:
 */

#ifndef FLOW_FUNC_META_MULTI_FUNC_H
#define FLOW_FUNC_META_MULTI_FUNC_H

#include <functional>
#include <map>
#include "flow_func_defines.h"
#include "meta_run_context.h"
#include "meta_params.h"
#include "flow_msg.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY MetaMultiFunc {
public:
    MetaMultiFunc() = default;

    virtual ~MetaMultiFunc() = default;

    /**
     * @brief multi func init.
     * @return 0:success, other: failed.
     */
    virtual int32_t Init(const std::shared_ptr<MetaParams> &params)
    {
        (void)params;
        return FLOW_FUNC_SUCCESS;
    }
};

using PROC_FUNC_WITH_CONTEXT =
    std::function<int32_t(const std::shared_ptr<MetaRunContext> &, const std::vector<std::shared_ptr<FlowMsg>> &)>;

using MULTI_FUNC_CREATOR_FUNC = std::function<int32_t(
    std::shared_ptr<MetaMultiFunc> &multiFunc, std::map<AscendString, PROC_FUNC_WITH_CONTEXT> &procFuncList)>;

/**
 * @brief register multi func creator.
 * @param flowFuncName cannot be null and must end with '\0'
 * @param func_creator multi func creator
 * @return if register success, true:success.
 */
FLOW_FUNC_VISIBILITY bool RegisterMultiFunc(
    const char *flowFuncName, const MULTI_FUNC_CREATOR_FUNC &funcCreator) noexcept;

template <typename T>
class FlowFuncRegistrar {
public:
    using CUSTOM_PROC_FUNC = std::function<int32_t(
        T *, const std::shared_ptr<MetaRunContext> &, const std::vector<std::shared_ptr<FlowMsg>> &)>;

    FlowFuncRegistrar &RegProcFunc(const char *flowFuncName, const CUSTOM_PROC_FUNC &func)
    {
        using namespace std::placeholders;
        funcMap_[flowFuncName] = func;
        (void)RegisterMultiFunc(flowFuncName, std::bind(&FlowFuncRegistrar::CreateMultiFunc, this, _1, _2));
        return *this;
    }

    int32_t CreateMultiFunc(std::shared_ptr<MetaMultiFunc> &multiFunc,
        std::map<AscendString, PROC_FUNC_WITH_CONTEXT> &procFuncMap) const
    {
        using namespace std::placeholders;
        T *flowFuncPtr = new(std::nothrow) T();
        if (flowFuncPtr == nullptr) {
            return FLOW_FUNC_FAILED;
        }
        multiFunc.reset(flowFuncPtr);
        for (const auto &func : funcMap_) {
            procFuncMap[func.first] = std::bind(func.second, flowFuncPtr, _1, _2);
        }
        return FLOW_FUNC_SUCCESS;
    }

private:
    std::map<AscendString, CUSTOM_PROC_FUNC> funcMap_;
};
/**
 * @brief define flow func REGISTRAR.
 * example:
 * FLOW_FUNC_REGISTRAR(UserFlowFunc).RegProcFunc("xxx_func", &UserFlowFunc::Proc1).
 *         RegProcFunc("xxx_func", &UserFlowFunc::Proc2);
 */
#define FLOW_FUNC_REGISTRAR(clazz)                                          \
    static FlowFunc::FlowFuncRegistrar<clazz> g_##clazz##FlowFuncRegistrar; \
    static auto &g_##clazz##Registrar = g_##clazz##FlowFuncRegistrar
}  // namespace FlowFunc
#endif  // FLOW_FUNC_META_MULTI_FUNC_H
