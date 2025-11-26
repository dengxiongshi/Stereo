/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2019-2023. All rights reserved.
 * Description: ot_common_adec.h
 * Author: Hisilicon multimedia software group
 * Create: 2019/06/15
 */

#ifndef  OT_COMMON_ADEC_H
#define  OT_COMMON_ADEC_H

#include "ot_common_aio.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

#define OT_MAX_DECODER_NAME_LEN        17

typedef struct {
    hi_u32 reserved;
} ot_adec_attr_g711;

typedef enum {
    OT_ADEC_MODE_PACK = 0, /* require input is valid dec pack(a
                              complete frame encode result),
                              e.g.the stream get from AENC is a
                              valid dec pack, the stream know actually
                              pack len from file is also a dec pack.
                              this mode is high-performative */
    OT_ADEC_MODE_STREAM,  /* input is stream, low-performative,
                             if you couldn't find out whether a stream is
                             valid dec pack,you could use
                             this mode */
    OT_ADEC_MODE_BUTT
} ot_adec_mode;

typedef struct {
    ot_payload_type    type;
    hi_u32             buf_size;   /* buf size[2~OT_MAX_ADEC_AENC_FRAME_NUM] */
    ot_adec_mode       mode;       /* decode mode */
    hi_void ATTRIBUTE *value;
} ot_adec_chn_attr;

typedef struct {
    hi_bool end_of_stream;      /* EOS flag */
    hi_u32  buf_total_num;   /* total number of channel buffer */
    hi_u32  buf_free_num;    /* free number of channel buffer */
    hi_u32  buf_busy_num;    /* busy number of channel buffer */
} ot_adec_chn_state;

typedef struct {
    ot_payload_type type;
    hi_char         name[OT_MAX_DECODER_NAME_LEN];

    hi_s32  (*func_open_decoder)(hi_void *decoder_attr, hi_void **decoder); /* struct decoder is packed by user,
                                                                               user malloc and free memory
                                                                               for this struct */
    hi_s32  (*func_dec_frame)(hi_void *decoder, hi_u8 **in_buf, hi_s32 *left_byte,
                              hi_u16 *out_buf, hi_u32 *out_len, hi_u32 *chns);
    hi_s32  (*func_get_frame_info)(hi_void *decoder, hi_void *info);
    hi_s32  (*func_close_decoder)(hi_void *decoder);
    hi_s32  (*func_reset_decoder)(hi_void *decoder);
} ot_adec_decoder;

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */

#endif /* OT_COMMON_ADEC_H */

