/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 * Description:
 */
#ifndef FLOW_FUNC_ATTR_VALUE_H
#define FLOW_FUNC_ATTR_VALUE_H

#include <vector>
#include <memory>
#include "flow_func_defines.h"
#include "tensor_data_type.h"
#include "ascend_string.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY AttrValue {
public:
    AttrValue() = default;

    virtual ~AttrValue() = default;

    AttrValue(const AttrValue &) = delete;

    AttrValue(AttrValue &&) = delete;

    AttrValue &operator=(const AttrValue &) = delete;

    AttrValue &operator=(AttrValue &&) = delete;

    /*
     * get string value of attr.
     * @param value: string value of attr
     * @return 0:SUCCESS, other:failed
     */
    virtual int32_t GetVal(AscendString &value) const = 0;

    /*
     * get string list value of attr.
     * @param value: string list value of attr
     * @return 0:SUCCESS, other:failed
     */
    virtual int32_t GetVal(std::vector<AscendString> &value) const = 0;

    /*
     * get int value of attr.
     * @param value: int value of attr
     * @return 0:SUCCESS, other:failed
     */
    virtual int32_t GetVal(int64_t &value) const = 0;

    /*
     * get int list value of attr.
     * @param value: int list value of attr
     * @return 0:SUCCESS, other:failed
     */
    virtual int32_t GetVal(std::vector<int64_t> &value) const = 0;

    /*
     * get int list list value of attr.
     * @param value: int list list value of attr
     * @return 0:SUCCESS, other:failed
     */
    virtual int32_t GetVal(std::vector<std::vector<int64_t>> &value) const = 0;

    /*
     * get float value of attr.
     * @param value: float value of attr
     * @return 0:SUCCESS, other:failed
     */
    virtual int32_t GetVal(float &value) const = 0;

    /*
     * get float list value of attr.
     * @param value: float list value of attr
     * @return 0:SUCCESS, other:failed
     */
    virtual int32_t GetVal(std::vector<float> &value) const = 0;

    /*
     * get bool value of attr.
     * @param value: bool value of attr
     * @return 0:SUCCESS, other:failed
     */
    virtual int32_t GetVal(bool &value) const = 0;

    /*
     * get bool list value of attr.
     * @param value: bool list value of attr
     * @return 0:SUCCESS, other:failed
     */
    virtual int32_t GetVal(std::vector<bool> &value) const = 0;

    /*
     * get data type value of attr.
     * @param value: data type value of attr
     * @return 0:SUCCESS, other:failed
     */
    virtual int32_t GetVal(TensorDataType &value) const = 0;

    /*
     * get data type list value of attr.
     * @param value: data type list value of attr
     * @return 0:SUCCESS, other:failed
     */
    virtual int32_t GetVal(std::vector<TensorDataType> &value) const = 0;
};
}

#endif // FLOW_FUNC_ATTR_VALUE_H
