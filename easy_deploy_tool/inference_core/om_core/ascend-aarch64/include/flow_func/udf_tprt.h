/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 * Description:
 */
#ifndef FLOW_FUNC_TPRT_H
#define FLOW_FUNC_TPRT_H

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include "meta_params.h"

namespace FlowFunc {
enum class UdfTprtTaskPriority {
    LOW = 0,
    NORMAL,
    HIGH,
    MAX,
};

class UdfTprtTaskAttrImpl;
class FLOW_FUNC_VISIBILITY UdfTprtTaskAttr {
public:
    /**
     * @brief Construct a task attr object
     */
    UdfTprtTaskAttr();

    /**
     * @brief Destory a task attr object
     */
    ~UdfTprtTaskAttr();

    /**
     * @brief Set task name
     */
    UdfTprtTaskAttr &SetName(const char *name);

    /**
     * @brief Get task name
     */
    const char *GetName() const;

    /**
     * @brief Set priority
     */
    UdfTprtTaskAttr &SetPriority(uint8_t priority);

    /**
     * @brief Get priority
     */
    uint8_t GetPriority() const;
private:
    std::shared_ptr<UdfTprtTaskAttrImpl> impl_;
};

class FLOW_FUNC_VISIBILITY UdfTprt {
public:

    /**
     * @brief tprt init.
     * @param param init param.
     * @param reserve reserve for extend.
     * @return 0:success, other failed.
     */
    static int32_t Init(const std::shared_ptr<MetaParams> &params);

    /**
     * @brief tprt submit.
     * @param func task func.
     * @param inDeps task in dependence.
     * @param outDeps task out dependence.
     * @param attr task attr.
     * @return 0:success, other failed.
     */
    static int32_t Submit(std::function<void()> &&func, const std::vector<const void*> &inDeps,
                          const std::vector<const void*> &outDeps, const UdfTprtTaskAttr &attr);

    /**
     * @brief tprt wait.
     * @return 0:success, other failed.
     */
    static int32_t Wait();

    /**
     * @brief get total thread num.
     * @return total thread num.
     */
    static int32_t GetNumThreads();

    /**
     * @brief get current thread id.
     * @return current thread id.
     */
    static int32_t GetThreadNum();

    /**
     * @brief parallel interface.
     * @param first fisrt index, included.
     * @param last last index, excluded.
     * @param step split step size.
     * @param func task func.
     * @return 0:success, other failed.
     */
    static void ParallelFor(const int64_t first, const int64_t last, const int64_t step,
                            const std::function<void(const int64_t index)> &func);

    /**
     * @brief parallel interface.
     * @param first fisrt index, included.
     * @param last last index, excluded.
     * @param func task func.
     * @return 0:success, other failed.
     */
    static void ParallelFor(const int64_t first, const int64_t last,
                            const std::function<void(const int64_t index)> &func);
};
}
#endif // FLOW_FUNC_TPRT_H