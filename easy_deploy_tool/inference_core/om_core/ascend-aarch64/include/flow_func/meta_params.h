/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 * Description:
 */
#ifndef FLOW_FUNC_META_PARAMS_H
#define FLOW_FUNC_META_PARAMS_H

#include <vector>
#include <memory>
#include "flow_func_defines.h"
#include "attr_value.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY MetaParams {
public:
    MetaParams() = default;

    virtual ~MetaParams() = default;

    virtual const char *GetName() const = 0;

    /**
     * @brief get attr.
     * @param attrName attr name, cannot be null, must end with '\0'.
     * @return AttrValue *: not null->success, null->failed
     */
    virtual std::shared_ptr<const AttrValue> GetAttr(const char *attrName) const = 0;

    template<class T>
    int32_t GetAttr(const char *attrName, T &value) const
    {
        auto attrValue = GetAttr(attrName);
        if (attrValue == nullptr) {
            return FLOW_FUNC_ERR_ATTR_NOT_EXITS;
        }
        return attrValue->GetVal(value);
    }

    /**
     * @brief get flow func input num.
     * used for check whether the number of inputs is consistent.
     * @return input num.
     */
    virtual size_t GetInputNum() const = 0;

    /**
     * @brief get flow func output num.
     * used for check whether the number of outputs is consistent.
     * @return output num.
     */
    virtual size_t GetOutputNum() const = 0;

    /**
     * @brief get flow func work path.
     * used for check whether the number of outputs is consistent.
     * @return output num.
     */
    virtual const char *GetWorkPath() const = 0;

    /**
     * @brief get running device id.
     * @return device id.
     */
    virtual int32_t GetRunningDeviceId() const = 0;
};
}

#endif // FLOW_FUNC_META_PARAMS_H
