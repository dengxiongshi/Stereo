/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 * Description:
 */

#ifndef FLOW_FUNC_LOG_H
#define FLOW_FUNC_LOG_H

#include "flow_func_defines.h"

namespace FlowFunc {
enum class FLOW_FUNC_VISIBILITY FlowFuncLogType {
    DEBUG_LOG = 0,
    RUN_LOG = 1
};
enum class FLOW_FUNC_VISIBILITY FlowFuncLogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3
};

class FLOW_FUNC_VISIBILITY FlowFuncLogger {
public:
    static FlowFuncLogger &GetLogger(FlowFuncLogType type);

    static const char *GetLogExtHeader();

    FlowFuncLogger() = default;

    virtual ~FlowFuncLogger() = default;

    virtual bool IsLogEnable(FlowFuncLogLevel level) = 0;

    virtual void Error(const char *fmt, ...) __attribute__((format(printf, 2, 3))) = 0;

    virtual void Warn(const char *fmt, ...) __attribute__((format(printf, 2, 3))) = 0;

    virtual void Info(const char *fmt, ...) __attribute__((format(printf, 2, 3))) = 0;

    virtual void Debug(const char *fmt, ...) __attribute__((format(printf, 2, 3))) = 0;
};
}

#define FLOW_FUNC_LOG_DEBUG(fmt, ...)                                               \
do {                                                                                \
    FlowFunc::FlowFuncLogger &debugLogger = FlowFunc::FlowFuncLogger::GetLogger(    \
        FlowFunc::FlowFuncLogType::DEBUG_LOG);                                      \
    if (debugLogger.IsLogEnable(FlowFunc::FlowFuncLogLevel::DEBUG)) {               \
        debugLogger.Debug("[%s:%d][%s]%s: " fmt, __FILE__, __LINE__, __FUNCTION__,  \
            FlowFunc::FlowFuncLogger::GetLogExtHeader(), ##__VA_ARGS__);            \
    }                                                                               \
} while (0)

#define FLOW_FUNC_LOG_INFO(fmt, ...)                                                \
do {                                                                                \
    FlowFunc::FlowFuncLogger &debugLogger = FlowFunc::FlowFuncLogger::GetLogger(    \
        FlowFunc::FlowFuncLogType::DEBUG_LOG);                                      \
    if (debugLogger.IsLogEnable(FlowFunc::FlowFuncLogLevel::INFO)) {                \
        debugLogger.Info("[%s:%d][%s]%s: " fmt, __FILE__, __LINE__, __FUNCTION__,   \
            FlowFunc::FlowFuncLogger::GetLogExtHeader(), ##__VA_ARGS__);            \
    }                                                                               \
} while (0)

#define FLOW_FUNC_LOG_WARN(fmt, ...)                                                \
do {                                                                                \
    FlowFunc::FlowFuncLogger &debugLogger = FlowFunc::FlowFuncLogger::GetLogger(    \
        FlowFunc::FlowFuncLogType::DEBUG_LOG);                                      \
    if (debugLogger.IsLogEnable(FlowFunc::FlowFuncLogLevel::WARN)) {                \
        debugLogger.Warn("[%s:%d][%s]%s: " fmt, __FILE__, __LINE__, __FUNCTION__,   \
            FlowFunc::FlowFuncLogger::GetLogExtHeader(), ##__VA_ARGS__);            \
    }                                                                               \
} while (0)

#define FLOW_FUNC_LOG_ERROR(fmt, ...)                                                   \
do {                                                                                    \
    FlowFunc::FlowFuncLogger::GetLogger(FlowFunc::FlowFuncLogType::DEBUG_LOG).Error(    \
        "[%s:%d][%s]%s: " fmt, __FILE__, __LINE__, __FUNCTION__,                        \
        FlowFunc::FlowFuncLogger::GetLogExtHeader(), ##__VA_ARGS__);                    \
} while (0)

#define FLOW_FUNC_RUN_LOG_INFO(fmt, ...)                                            \
do {                                                                                \
    FlowFunc::FlowFuncLogger &runLogger = FlowFunc::FlowFuncLogger::GetLogger(      \
        FlowFunc::FlowFuncLogType::RUN_LOG);                                        \
    if (runLogger.IsLogEnable(FlowFunc::FlowFuncLogLevel::INFO)) {                  \
        runLogger.Info("[%s:%d][%s]%s: " fmt, __FILE__, __LINE__, __FUNCTION__,     \
            FlowFunc::FlowFuncLogger::GetLogExtHeader(), ##__VA_ARGS__);            \
    }                                                                               \
} while (0)

#define FLOW_FUNC_RUN_LOG_ERROR(fmt, ...)                                           \
do {                                                                                \
    FlowFunc::FlowFuncLogger::GetLogger(FlowFunc::FlowFuncLogType::RUN_LOG).Error(  \
        "[%s:%d][%s]%s: " fmt, __FILE__, __LINE__, __FUNCTION__,                    \
        FlowFunc::FlowFuncLogger::GetLogExtHeader(), ##__VA_ARGS__);                \
} while (0)
#endif // FLOW_FUNC_LOG_H
