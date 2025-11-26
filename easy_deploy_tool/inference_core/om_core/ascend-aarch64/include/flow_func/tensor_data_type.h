/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 * Description:
 */

#ifndef FLOW_FUNC_TENSOR_DATA_TYPE_H
#define FLOW_FUNC_TENSOR_DATA_TYPE_H

#include <cstdint>

namespace FlowFunc {
enum class TensorDataType : int32_t {
    DT_FLOAT = 0,           // float type
    DT_FLOAT16 = 1,         // fp16 type
    DT_INT8 = 2,            // int8 type
    DT_INT16 = 6,           // int16 type
    DT_UINT16 = 7,          // uint16 type
    DT_UINT8 = 4,           // uint8 type
    DT_INT32 = 3,           // int32 type
    DT_INT64 = 9,           // int64 type
    DT_UINT32 = 8,          // unsigned int32
    DT_UINT64 = 10,         // unsigned int64
    DT_BOOL = 12,           // bool type
    DT_DOUBLE = 11,         // double type
    DT_QINT8 = 18,          // qint8 type
    DT_QINT16 = 19,         // qint16 type
    DT_QINT32 = 20,         // qint32 type
    DT_QUINT8 = 21,         // quint8 type
    DT_QUINT16 = 22,        // quint16 type
    DT_DUAL = 25,           // dual output type
    DT_INT4 = 29,           // int4 type
    DT_UINT1 = 30,          // uint1 type
    DT_INT2 = 31,           // int2 type
    DT_UINT2 = 32,          // uint2 type
    DT_UNDEFINED            // Used to indicate a DataType field has not been set.
};
}  // namespace FlowFunc

#endif // FLOW_FUNC_TENSOR_DATA_TYPE_H
