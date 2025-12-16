/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: RemixV100 audio codec header
 * Author: Hisilicon multimedia software group
 * Create: 2022-09-10
 */

#ifndef OT_ACODEC_H
#define OT_ACODEC_H

#include "hi_media_type.h"
#include "hi_media_common.h"

#define IOC_TYPE_ACODEC 'A'

typedef enum {
    OT_ACODEC_FS_8000  = 0x1,
    OT_ACODEC_FS_11025 = 0x2,
    OT_ACODEC_FS_12000 = 0x3,
    OT_ACODEC_FS_16000 = 0x4,
    OT_ACODEC_FS_22050 = 0x5,
    OT_ACODEC_FS_24000 = 0x6,
    OT_ACODEC_FS_32000 = 0x7,
    OT_ACODEC_FS_44100 = 0x8,
    OT_ACODEC_FS_48000 = 0x9,
    OT_ACODEC_FS_64000 = 0xa,
    OT_ACODEC_FS_96000 = 0xb,
    OT_ACODEC_FS_BUTT = 0xc
} ot_acodec_fs;

typedef enum {
    OT_ACODEC_MIXER_IN0   = 0x0,
    OT_ACODEC_MIXER_IN1   = 0x1,
    OT_ACODEC_MIXER_IN_D  = 0x2,
    OT_ACODEC_MIXER_BUTT
} ot_acodec_mixer;

typedef struct {
    /* volume control, 0x00~0x7e, 0x7F:mute */
    hi_u32 volume_ctrl;
    /* adc/dac mute control, 1:mute, 0:unmute */
    hi_u32 volume_ctrl_mute;
} ot_acodec_volume_ctrl;

typedef enum {
    IOC_NR_SOFT_RESET_CTRL = 0x0,
    IOC_NR_SET_I2S1_FS,
    IOC_NR_SET_MIXER_MIC,

    /* input/output volume */
    IOC_NR_SET_INPUT_VOL,
    IOC_NR_SET_OUTPUT_VOL,
    IOC_NR_GET_INPUT_VOL,
    IOC_NR_GET_OUTPUT_VOL,

    /* analog part input gain */
    IOC_NR_BOOSTL_ENABLE,
    IOC_NR_BOOSTR_ENABLE,
    IOC_NR_SET_GAIN_MICL,
    IOC_NR_SET_GAIN_MICR,
    IOC_NR_GET_GAIN_MICL,
    IOC_NR_GET_GAIN_MICR,

    /* ADC/DAC volume */
    IOC_NR_SET_DACL_VOL,
    IOC_NR_SET_DACR_VOL,
    IOC_NR_SET_ADCL_VOL,
    IOC_NR_SET_ADCR_VOL,
    IOC_NR_GET_DACL_VOL,
    IOC_NR_GET_DACR_VOL,
    IOC_NR_GET_ADCL_VOL,
    IOC_NR_GET_ADCR_VOL
} acodec_ioc;

/* DAC volume control(left channel) ot_acodec_volume_ctrl */
#define OT_ACODEC_SET_DACL_VOLUME \
    _IOWR(IOC_TYPE_ACODEC, IOC_NR_SET_DACL_VOL, ot_acodec_volume_ctrl)
/* DAC volume control(right channel) ot_acodec_volume_ctrl */
#define OT_ACODEC_SET_DACR_VOLUME \
    _IOWR(IOC_TYPE_ACODEC, IOC_NR_SET_DACR_VOL, ot_acodec_volume_ctrl)
/* ADC volume control(left channel) ot_acodec_volume_ctrl */
#define OT_ACODEC_SET_ADCL_VOLUME \
    _IOWR(IOC_TYPE_ACODEC, IOC_NR_SET_ADCL_VOL, ot_acodec_volume_ctrl)
/* ADC volume control(right channel) ot_acodec_volume_ctrl */
#define OT_ACODEC_SET_ADCR_VOLUME \
    _IOWR(IOC_TYPE_ACODEC, IOC_NR_SET_ADCR_VOL, ot_acodec_volume_ctrl)

/* get DAC volume(left channel) */
#define OT_ACODEC_GET_DACL_VOLUME \
    _IOWR(IOC_TYPE_ACODEC, IOC_NR_GET_DACL_VOL, ot_acodec_volume_ctrl)
/* get DAC volume(right channel) */
#define OT_ACODEC_GET_DACR_VOLUME \
    _IOWR(IOC_TYPE_ACODEC, IOC_NR_GET_DACR_VOL, ot_acodec_volume_ctrl)
/* get ADC volume(left channel) */
#define OT_ACODEC_GET_ADCL_VOLUME \
    _IOWR(IOC_TYPE_ACODEC, IOC_NR_GET_ADCL_VOL, ot_acodec_volume_ctrl)
/* get ADC volume(right channel) */
#define OT_ACODEC_GET_ADCR_VOLUME \
    _IOWR(IOC_TYPE_ACODEC, IOC_NR_GET_ADCR_VOL, ot_acodec_volume_ctrl)

#endif /* OT_ACODEC_H */
