/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2021-2023. All rights reserved.
 * Description: multimedia common file
 * Author: Hisilicon multimedia software group
 * Create: 2021/04/27
 */

#ifndef HI_COMMON_AENC_H
#define HI_COMMON_AENC_H

#include "hi_common_aio.h"
#include "ot_common_aenc.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HI_AENC_MAX_CHN_NUM      OT_AENC_MAX_CHN_NUM
#define HI_MAX_ENCODER_NAME_LEN  OT_MAX_ENCODER_NAME_LEN

typedef ot_aenc_chn_attr         hi_aenc_chn_attr;
typedef ot_aenc_encoder          hi_aenc_encoder;
typedef ot_aenc_attr_g711        hi_aenc_attr_g711;

#ifdef __cplusplus
}
#endif
#endif /* HI_COMMON_AENC_H */
