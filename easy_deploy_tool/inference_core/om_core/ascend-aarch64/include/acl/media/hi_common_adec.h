/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2021-2023. All rights reserved.
 * Description: multimedia common file
 * Author: Hisilicon multimedia software group
 * Create: 2021/04/27
 */

#ifndef HI_COMMON_ADEC_H
#define HI_COMMON_ADEC_H

#include "hi_common_aio.h"
#include "ot_common_adec.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HI_ADEC_MAX_CHN_NUM     OT_ADEC_MAX_CHN_NUM
#define HI_MAX_DECODER_NAME_LEN OT_MAX_DECODER_NAME_LEN
#define HI_ADEC_MAX_CHN_NUM     OT_ADEC_MAX_CHN_NUM

#define HI_ADEC_MODE_PACK       OT_ADEC_MODE_PACK
#define HI_ADEC_MODE_STREAM     OT_ADEC_MODE_STREAM
#define HI_ADEC_MODE_BUTT       OT_ADEC_MODE_BUTT

typedef ot_adec_mode            hi_adec_mode;
typedef ot_adec_chn_attr        hi_adec_chn_attr;
typedef ot_adec_chn_state       hi_adec_chn_state;
typedef ot_adec_decoder         hi_adec_decoder;
typedef ot_audio_frame_info     hi_audio_frame_info;
typedef ot_adec_attr_g711       hi_adec_attr_g711;

#ifdef __cplusplus
}
#endif

#endif /* HI_COMMON_ADEC_H */
