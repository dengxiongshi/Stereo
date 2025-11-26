/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 * Description:
 */
#ifndef FLOW_FUNC_ASCEND_STRING_H
#define FLOW_FUNC_ASCEND_STRING_H

#include <string>
#include <memory>
#include "flow_func_defines.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY AscendString {
public:
    AscendString() = default;

    ~AscendString() = default;

    AscendString(const char * const name);

    AscendString(const char * const name, size_t length);

    /**
     * @brief get string info.
     * @return string
     */
    const char *GetString() const;

    size_t GetLength() const;

    /**
     * @brief check if current string is less than the target string.
     * @param the target string to be compared
     * @return true if src string is less than the target string, otherwise return false.
     */
    bool operator<(const AscendString &d) const;

    /**
     * @brief check if current string is larger than the target string.
     * @param the target string to be compared
     * @return true if src string is larger than the target string, otherwise return false.
     */
    bool operator>(const AscendString &d) const;

    /**
     * @brief check if current string is less than or equal to the target string.
     * @param the target string to be compared
     * @return true if src string is less than or equal to target string, otherwise return false.
     */
    bool operator<=(const AscendString &d) const;

    /**
     * @brief check if current string is larger than or equal to the target string.
     * @param the target string to be compared
     * @return true if src string is larger than or equal to the target string, otherwise return false.
     */
    bool operator>=(const AscendString &d) const;

    /**
     * @brief check if current string is equal to the target string.
     * @param the target string to be compared
     * @return true if src string is equal to the target string, otherwise return false.
     */
    bool operator==(const AscendString &d) const;

    /**
     * @brief check if current string is not equal to the target string.
     * @param the target string to be compared
     * @return true if src string is not equal to the target string, otherwise return false.
     */
    bool operator!=(const AscendString &d) const;

private:
    std::shared_ptr<std::string> name_;
};
}

#endif // FLOW_FUNC_ASCEND_STRING_H