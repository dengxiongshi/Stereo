/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 * Description:
 */
#ifndef FLOW_FUNC_META_CONTEXT_H
#define FLOW_FUNC_META_CONTEXT_H

#include <vector>
#include <memory>
#include "flow_func_defines.h"
#include "attr_value.h"
#include "flow_msg.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY MetaContext {
public:
    MetaContext() = default;

    virtual ~MetaContext() = default;

    /**
     * @brief alloc tensor msg by shape and data type.
     * @param shape tensor shape
     * @param dataType data type
     * @return tensor
     */
    virtual std::shared_ptr<FlowMsg> AllocTensorMsg(const std::vector<int64_t> &shape, TensorDataType dataType) = 0;

    /**
     * @brief set output tensor.
     * @param outIdx output index, start from 0.
     * @param outMsg output msg.
     * @return 0:success, other failed.
     */
    virtual int32_t SetOutput(uint32_t outIdx, std::shared_ptr<FlowMsg> outMsg) = 0;

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
     * @brief alloc empty data msg.
     * @param msgType msg type which msg will be alloc
     * @return empty data FlowMsg
     */
    virtual std::shared_ptr<FlowMsg> AllocEmptyDataMsg(MsgType msgType) = 0;

    /**
     * @brief run flow model.
     * @param modelKey invoked flow model key.
     * @param inputMsgs flow model input message.
     * @param outputMsgs flow model output message.
     * @param timeout timeout(ms), -1 means never timeout.
     * @return 0:success, other failed.
     */
    virtual int32_t RunFlowModel(const char *modelKey, const std::vector<std::shared_ptr<FlowMsg>> &inputMsgs,
        std::vector<std::shared_ptr<FlowMsg>> &outputMsgs, int32_t timeout) = 0;

    /**
     * @brief get flow func work path.
     * @return flow func work path.
     */
    virtual const char *GetWorkPath() const = 0;

    /**
     * @brief get running device id.
     * @return device id.
     */
    virtual int32_t GetRunningDeviceId() const = 0;

    /**
     * @brief get user data, max data size is 64.
     * @param data user data point, output.
     * @param size user data size, need in (0, 64].
     * @param offset user data offset, need in [0, 64), size + offset <= 64.
     * @return success:FLOW_FUNC_SUCCESS, failed:OTHERS.
     */
    virtual int32_t GetUserData(void *data, size_t size, size_t offset = 0U) const = 0;
};
}

#endif // FLOW_FUNC_META_CONTEXT_H
