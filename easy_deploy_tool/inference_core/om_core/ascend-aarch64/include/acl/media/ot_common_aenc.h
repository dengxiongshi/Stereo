/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2019-2023. All rights reserved.
 * Description: ot_common_aenc.h
 * Author: Hisilicon multimedia software group
 * Create: 2019/06/15
 */

#ifndef  OT_COMMON_AENC_H
#define  OT_COMMON_AENC_H

#include "ot_common_aio.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

#define OT_MAX_ENCODER_NAME_LEN    17

typedef struct {
    hi_u32 reserved;    /* reserve item */
} ot_aenc_attr_g711;

typedef struct {
    ot_payload_type type;
    hi_u32          max_frame_len;
    hi_char         name[OT_MAX_ENCODER_NAME_LEN];    /* encoder type,be used to print proc information */
    hi_s32          (*func_open_encoder)(hi_void *encoder_attr, hi_void **encoder); /* encoder is the handle to
                                                                                       control the encoder */
    hi_s32          (*func_enc_frame)(hi_void *encoder, const ot_audio_frame *data,
                                      hi_u8 *out_buf, hi_u32 *out_len);
    hi_s32          (*func_close_encoder)(hi_void *encoder);
} ot_aenc_encoder;

typedef struct {
    ot_payload_type     type;
    hi_u32              point_num_per_frame;
    hi_u32              buf_size;      /* buf size [2~OT_MAX_ADEC_AENC_FRAME_NUM] */
    hi_void ATTRIBUTE   *value;        /* point to attribute of definite audio encoder */
} ot_aenc_chn_attr;

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */

#endif /* OT_COMMON_AENC_H */

