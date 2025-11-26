/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 * Description:
 */
#ifndef FLOW_FUNC_FLOW_MSG_H
#define FLOW_FUNC_FLOW_MSG_H

#include <cstdint>
#include <vector>
#include "flow_func_defines.h"
#include "tensor_data_type.h"

namespace FlowFunc {
enum class MsgType : uint16_t {
    MSG_TYPE_TENSOR_DATA = 0,   // tensor data msg type
    MSG_TYPE_RAW_MSG            // raw data msg type
};

enum class FlowFlag : uint32_t {
    FLOW_FLAG_EOS = (1U << 0U),       // data flag end
    FLOW_FLAG_SEG = (1U << 1U)  // segment flag for discontinuous
};

class FLOW_FUNC_VISIBILITY Tensor {
public:
    Tensor() = default;

    virtual ~Tensor() = default;

    virtual const std::vector<int64_t> &GetShape() const = 0;

    virtual TensorDataType GetDataType() const = 0;

    virtual void *GetData() const = 0;

    virtual uint64_t GetDataSize() const = 0;

    /**
     * @brief get tensor element count.
     * @return element count. if return -1, it means unknown or overflow.
     */
    virtual int64_t GetElementCnt() const = 0;
};

class FLOW_FUNC_VISIBILITY FlowMsg {
public:
    FlowMsg() = default;

    virtual ~FlowMsg() = default;

    virtual MsgType GetMsgType() const = 0;

    /**
     * @brief get tensor.
     * when MsgType=TENSOR_DATA_TYPE, can get tensor.
     * @return tensor.
     * attention:
     * Tensor life cycle is same as FlowMsg, Caller should not keep it alone.
     */
    virtual Tensor *GetTensor() const = 0;

    virtual int32_t GetRetCode() const = 0;

    virtual void SetRetCode(int32_t retCode) = 0;

    virtual void SetStartTime(uint64_t startTime) = 0;

    virtual uint64_t GetStartTime() const = 0;

    virtual void SetEndTime(uint64_t endTime) = 0;

    virtual uint64_t GetEndTime() const = 0;

    /**
     * @brief set flow flags.
     * @param flags flags is FlowFlag set, multi FlowFlags can use bit or operation.
     */
    virtual void SetFlowFlags(uint32_t flags) = 0;

    /**
     * @brief get flow flags.
     * @return FlowFlag set, can use bit and operation check if FlowFlag is set.
     */
    virtual uint32_t GetFlowFlags() const = 0;

    /**
     * @brief set route label.
     * @param routeLabel route label, value 0 means not used.
     */
    virtual void SetRouteLabel(uint32_t routeLabel) = 0;
};
}

#endif // FLOW_FUNC_FLOW_MSG_H