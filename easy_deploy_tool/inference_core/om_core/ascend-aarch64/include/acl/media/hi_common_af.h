/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: Api of af
 * Author: Hisilicon multimedia software group
 * Create: 2023-01-05
 */
#ifndef HI_COMMON_AF_H
#define HI_COMMON_AF_H

#include "hi_media_type.h"
#include "hi_common_isp.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

#define HI_AF_LIB_NAME "hisi_af_lib"

typedef struct {
    hi_s32  max_focus_pos;      /* far */
    hi_s32  min_focus_pos;      /* near */
} hi_isp_af_sensor_default;


typedef struct {
    hi_s32 (*pfn_cmos_get_af_default)(hi_vi_pipe vi_pipe, hi_isp_af_sensor_default *af_sns_dft);
} hi_isp_af_sensor_exp_func;

typedef struct {
    hi_isp_af_sensor_exp_func sns_exp;
} hi_isp_af_sensor_register;

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */

#endif
