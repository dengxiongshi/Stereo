/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: multimedia common file
 * Author: Hisilicon multimedia software group
 * Create: 2022/04/27
 */

#ifndef HI_ACODEC_H
#define HI_ACODEC_H

#include "ot_acodec.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef ot_acodec_volume_ctrl     hi_acodec_volume_ctrl;

#define HI_ACODEC_SET_DACL_VOLUME OT_ACODEC_SET_DACL_VOLUME
#define HI_ACODEC_SET_DACR_VOLUME OT_ACODEC_SET_DACR_VOLUME
#define HI_ACODEC_SET_ADCL_VOLUME OT_ACODEC_SET_ADCL_VOLUME
#define HI_ACODEC_SET_ADCR_VOLUME OT_ACODEC_SET_ADCR_VOLUME
#define HI_ACODEC_GET_DACL_VOLUME OT_ACODEC_GET_DACL_VOLUME
#define HI_ACODEC_GET_DACR_VOLUME OT_ACODEC_GET_DACR_VOLUME
#define HI_ACODEC_GET_ADCL_VOLUME OT_ACODEC_GET_ADCL_VOLUME
#define HI_ACODEC_GET_ADCR_VOLUME OT_ACODEC_GET_ADCR_VOLUME

#ifdef __cplusplus
}
#endif
#endif /* __HI_ACODEC_H__ */
