/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2021-2023. All rights reserved.
 * Description: multimedia common file
 * Author: Hisilicon multimedia software group
 * Create: 2021/04/27
 */

#ifndef HI_COMMON_AIO_H
#define HI_COMMON_AIO_H

#include "ot_common_aio.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HI_AIO_MAX_NUM                  OT_AIO_MAX_NUM
#define HI_AI_DEV_MAX_NUM               OT_AI_DEV_MAX_NUM
#define HI_AO_DEV_MAX_NUM               OT_AO_DEV_MAX_NUM

#define HI_AIO_MAX_CHN_NUM              OT_AIO_MAX_CHN_NUM
#define HI_AI_MAX_CHN_NUM               OT_AI_MAX_CHN_NUM
#define HI_AO_MAX_CHN_NUM               OT_AO_MAX_CHN_NUM
#define HI_AO_SYS_CHN_ID                OT_AO_SYS_CHN_ID

#define HI_MAX_AUDIO_FILE_PATH_LEN      OT_MAX_AUDIO_FILE_PATH_LEN
#define HI_MAX_AUDIO_FILE_NAME_LEN      OT_MAX_AUDIO_FILE_NAME_LEN

#define HI_AUDIO_FRAME_CHN_NUM          OT_AUDIO_FRAME_CHN_NUM
#define HI_MAX_AUDIO_FRAME_LEN          OT_MAX_AUDIO_FRAME_LEN
#define HI_MAX_AUDIO_STREAM_LEN         OT_MAX_AUDIO_STREAM_LEN

#define HI_MAX_AUDIO_FRAME_NUM          OT_MAX_AUDIO_FRAME_NUM
#define HI_MAX_ADEC_AENC_FRAME_NUM      OT_MAX_ADEC_AENC_FRAME_NUM
#define HI_MAX_AUDIO_POINT_BYTES        OT_MAX_AUDIO_POINT_BYTES
#define HI_MAX_VOICE_POINT_NUM          OT_MAX_VOICE_POINT_NUM
#define HI_MAX_AUDIO_POINT_NUM          OT_MAX_AUDIO_POINT_NUM
#define HI_MAX_AO_POINT_NUM             OT_MAX_AO_POINT_NUM
#define HI_MIN_AUDIO_POINT_NUM          OT_MIN_AUDIO_POINT_NUM

// audio sample rate defines
#define HI_AUDIO_SAMPLE_RATE_8000       OT_AUDIO_SAMPLE_RATE_8000
#define HI_AUDIO_SAMPLE_RATE_12000      OT_AUDIO_SAMPLE_RATE_12000
#define HI_AUDIO_SAMPLE_RATE_11025      OT_AUDIO_SAMPLE_RATE_11025
#define HI_AUDIO_SAMPLE_RATE_16000      OT_AUDIO_SAMPLE_RATE_16000
#define HI_AUDIO_SAMPLE_RATE_22050      OT_AUDIO_SAMPLE_RATE_22050
#define HI_AUDIO_SAMPLE_RATE_24000      OT_AUDIO_SAMPLE_RATE_24000
#define HI_AUDIO_SAMPLE_RATE_32000      OT_AUDIO_SAMPLE_RATE_32000
#define HI_AUDIO_SAMPLE_RATE_44100      OT_AUDIO_SAMPLE_RATE_44100
#define HI_AUDIO_SAMPLE_RATE_48000      OT_AUDIO_SAMPLE_RATE_48000
#define HI_AUDIO_SAMPLE_RATE_64000      OT_AUDIO_SAMPLE_RATE_64000
#define HI_AUDIO_SAMPLE_RATE_96000      OT_AUDIO_SAMPLE_RATE_96000
#define HI_AUDIO_SAMPLE_RATE_BUTT       OT_AUDIO_SAMPLE_RATE_BUTT
typedef ot_audio_sample_rate            hi_audio_sample_rate;

// audio mode defines
#define HI_AIO_MODE_I2S_MASTER         OT_AIO_MODE_I2S_MASTER
#define HI_AIO_MODE_I2S_SLAVE          OT_AIO_MODE_I2S_SLAVE
#define HI_AIO_MODE_PCM_SLAVE_STD      OT_AIO_MODE_PCM_SLAVE_STD
#define HI_AIO_MODE_PCM_SLAVE_NON_STD  OT_AIO_MODE_PCM_SLAVE_NON_STD
#define HI_AIO_MODE_PCM_MASTER_STD     OT_AIO_MODE_PCM_MASTER_STD
#define HI_AIO_MODE_PCM_MASTER_NON_STD OT_AIO_MODE_PCM_MASTER_NON_STD
#define HI_AIO_MODE_BUTT               OT_AIO_MODE_BUTT
typedef ot_aio_mode                    hi_aio_mode;

// audio bit width defines
#define HI_AUDIO_BIT_WIDTH_8            OT_AUDIO_BIT_WIDTH_8
#define HI_AUDIO_BIT_WIDTH_16           OT_AUDIO_BIT_WIDTH_16
#define HI_AUDIO_BIT_WIDTH_24           OT_AUDIO_BIT_WIDTH_24
#define HI_AUDIO_BIT_WIDTH_BUTT         OT_AUDIO_BIT_WIDTH_BUTT
typedef ot_audio_bit_width              hi_audio_bit_width;

// audio sound mode defines
#define HI_AUDIO_SOUND_MODE_MONO        OT_AUDIO_SOUND_MODE_MONO
#define HI_AUDIO_SOUND_MODE_STEREO      OT_AUDIO_SOUND_MODE_STEREO
#define HI_AUDIO_SOUND_MODE_BUTT        OT_AUDIO_SOUND_MODE_BUTT
typedef ot_audio_snd_mode               hi_audio_snd_mode;

// audio i2s type defines
#define HI_AIO_I2STYPE_INNERCODEC       OT_AIO_I2STYPE_INNERCODEC
#define HI_AIO_I2STYPE_INNERHDMI        OT_AIO_I2STYPE_INNERHDMI
#define HI_AIO_I2STYPE_EXTERN           OT_AIO_I2STYPE_EXTERN
typedef ot_aio_i2s_type                 hi_aio_i2s_type;

// audio ai channel mode and attribute
#define HI_AI_CHN_MODE_NORMAL           OT_AI_CHN_MODE_NORMAL
#define HI_AI_CHN_MODE_FAST             OT_AI_CHN_MODE_FAST
#define HI_AI_CHN_MODE_BUTT             OT_AI_CHN_MODE_BUTT
typedef ot_ai_chn_mode                  hi_ai_chn_mode;
typedef ot_ai_chn_attr                  hi_ai_chn_attr;

typedef ot_aio_attr                     hi_aio_attr;
typedef ot_audio_frame                  hi_audio_frame;
typedef ot_audio_stream                 hi_audio_stream;
typedef ot_aec_frame                    hi_aec_frame;

#ifdef __cplusplus
}
#endif
#endif /* __HI_EXT_COMMON_AIO_H__ */
