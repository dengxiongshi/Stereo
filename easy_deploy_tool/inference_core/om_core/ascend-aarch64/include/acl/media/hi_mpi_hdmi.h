/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: hdmi common file
 * Author: Hisilicon multimedia software group
 * Create: 2023/03/19
 */

#ifndef HI_MPI_HDMI_H
#define HI_MPI_HDMI_H

#include <stdint.h>
#include "hi_media_type.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HI_HDMI_ID_0,
    HI_HDMI_ID_1,
    HI_HDMI_ID_BUTT
} hi_hdmi_id;

typedef enum {
    HI_HDMI_VIDEO_FORMAT_1080P_60,
    HI_HDMI_VIDEO_FORMAT_1080P_50,
    HI_HDMI_VIDEO_FORMAT_1080P_30,
    HI_HDMI_VIDEO_FORMAT_1080P_25,
    HI_HDMI_VIDEO_FORMAT_1080P_24,
    HI_HDMI_VIDEO_FORMAT_1080i_60,
    HI_HDMI_VIDEO_FORMAT_1080i_50,
    HI_HDMI_VIDEO_FORMAT_720P_60,
    HI_HDMI_VIDEO_FORMAT_720P_50,
    HI_HDMI_VIDEO_FORMAT_576P_50,
    HI_HDMI_VIDEO_FORMAT_480P_60,
    HI_HDMI_VIDEO_FORMAT_PAL,
    HI_HDMI_VIDEO_FORMAT_NTSC,
    HI_HDMI_VIDEO_FORMAT_861D_640X480_60,
    HI_HDMI_VIDEO_FORMAT_VESA_800X600_60,
    HI_HDMI_VIDEO_FORMAT_VESA_1024X768_60,
    HI_HDMI_VIDEO_FORMAT_VESA_1280X800_60,
    HI_HDMI_VIDEO_FORMAT_VESA_1280X1024_60,
    HI_HDMI_VIDEO_FORMAT_VESA_1366X768_60,
    HI_HDMI_VIDEO_FORMAT_VESA_1440X900_60,
    HI_HDMI_VIDEO_FORMAT_VESA_1400X1050_60,
    HI_HDMI_VIDEO_FORMAT_VESA_1600X1200_60,
    HI_HDMI_VIDEO_FORMAT_VESA_1680X1050_60,
    HI_HDMI_VIDEO_FORMAT_VESA_1920X1200_60,
    HI_HDMI_VIDEO_FORMAT_2560x1440_30,
    HI_HDMI_VIDEO_FORMAT_2560x1440_60,
    HI_HDMI_VIDEO_FORMAT_2560x1600_60,
    HI_HDMI_VIDEO_FORMAT_1920x2160_30,
    HI_HDMI_VIDEO_FORMAT_3840X2160P_24,
    HI_HDMI_VIDEO_FORMAT_3840X2160P_25,
    HI_HDMI_VIDEO_FORMAT_3840X2160P_30,
    HI_HDMI_VIDEO_FORMAT_3840X2160P_50,
    HI_HDMI_VIDEO_FORMAT_3840X2160P_60,
    HI_HDMI_VIDEO_FORMAT_4096X2160P_24,
    HI_HDMI_VIDEO_FORMAT_4096X2160P_25,
    HI_HDMI_VIDEO_FORMAT_4096X2160P_30,
    HI_HDMI_VIDEO_FORMAT_4096X2160P_50,
    HI_HDMI_VIDEO_FORMAT_4096X2160P_60,
    HI_HDMI_VIDEO_FORMAT_3840X2160P_120,
    HI_HDMI_VIDEO_FORMAT_4096X2160P_120,
    HI_HDMI_VIDEO_FORMAT_7680X4320P_30,
    HI_HDMI_VIDEO_FORMAT_VESA_CUSTOMER_DEFINE,
    HI_HDMI_VIDEO_FORMAT_BUTT
} hi_hdmi_video_format;

typedef enum {
    HI_HDMI_DEEP_COLOR_24BIT, /* HDMI Deep Color 24bit mode */
    HI_HDMI_DEEP_COLOR_30BIT, /* HDMI Deep Color 30bit mode */
    HI_HDMI_DEEP_COLOR_36BIT, /* HDMI Deep Color 36bit mode */
    HI_HDMI_DEEP_COLOR_BUTT
} hi_hdmi_deep_color;

typedef enum {
    HI_HDMI_SAMPLE_RATE_UNKNOWN, /* unknown sample rate */
    HI_HDMI_SAMPLE_RATE_8K,      /* 8K sample rate */
    HI_HDMI_SAMPLE_RATE_11K,     /* 11.025K sample rate */
    HI_HDMI_SAMPLE_RATE_12K,     /* 12K sample rate */
    HI_HDMI_SAMPLE_RATE_16K,     /* 16K sample rate */
    HI_HDMI_SAMPLE_RATE_22K,     /* 22.050K sample rate */
    HI_HDMI_SAMPLE_RATE_24K,     /* 24K sample rate */
    HI_HDMI_SAMPLE_RATE_32K,     /* 32K sample rate */
    HI_HDMI_SAMPLE_RATE_44K,     /* 44.1K sample rate */
    HI_HDMI_SAMPLE_RATE_48K,     /* 48K sample rate */
    HI_HDMI_SAMPLE_RATE_88K,     /* 88.2K sample rate */
    HI_HDMI_SAMPLE_RATE_96K,     /* 96K sample rate */
    HI_HDMI_SAMPLE_RATE_176K,    /* 176K sample rate */
    HI_HDMI_SAMPLE_RATE_192K,    /* 192K sample rate */
    HI_HDMI_SAMPLE_RATE_768K,    /* 768K sample rate */
    HI_HDMI_SAMPLE_RATE_BUTT
} hi_hdmi_sample_rate;

typedef enum {
    HI_HDMI_BIT_DEPTH_UNKNOWN, /* unknown bit width */
    HI_HDMI_BIT_DEPTH_8,       /* 8 bits width */
    HI_HDMI_BIT_DEPTH_16,      /* 16 bits width */
    HI_HDMI_BIT_DEPTH_18,      /* 18 bits width */
    HI_HDMI_BIT_DEPTH_20,      /* 20 bits width */
    HI_HDMI_BIT_DEPTH_24,      /* 24 bits width */
    HI_HDMI_BIT_DEPTH_32,      /* 32 bits width */
    HI_HDMI_BIT_DEPTH_BUTT
} hi_hdmi_bit_depth;

typedef struct {
    /* Whether to forcibly output the video over the HDMI. */
    hi_bool hdmi_en;
    /* Video norm. This value of the video norm must be consistent with the norm of the video output. */
    hi_hdmi_video_format video_format;
    /* DeepColor output mode.It is OT_HDMI_DEEP_COLOR_24BIT by default. */
    hi_hdmi_deep_color deep_color_mode;
    /* Whether to enable the audio. */
    hi_bool audio_en;
    /* Audio sampling rate. This parameter needs to be consistent with that of the VO. */
    hi_hdmi_sample_rate sample_rate;
    /* Audio bit width. It is 16 by default. This parameter needs to be consistent with that of the VO. */
    hi_hdmi_bit_depth bit_depth;
    /* Whether to enable auth mode. 0: disabled 1: enabled */
    hi_bool auth_mode_en;
    /*
     * Enable flag of deep color mode adapting case of user setting incorrect,
     * default: TD_FALSE.When user have no any adapting strategy,suggestion TD_TRUE
     */
    hi_bool deep_color_adapt_en;
    /*
     * Pixclk of enVideoFmt(unit is kHz).
     * (This param is valid only when enVideoFmt is OT_HDMI_VIDEO_FMT_VESA_CUSTOMER_DEFINE)
     */
    hi_u32 pix_clk;
} hi_hdmi_attr;

typedef enum {
    HI_INFOFRAME_TYPE_AVI,
    HI_INFOFRAME_TYPE_AUDIO,
    HI_INFOFRAME_TYPE_VENDORSPEC,
    HI_INFOFRAME_TYPE_BUTT
} hi_hdmi_infoframe_type;

typedef enum {
    HI_HDMI_COLOR_SPACE_RGB444,
    HI_HDMI_COLOR_SPACE_YCBCR422,
    HI_HDMI_COLOR_SPACE_YCBCR444,
    /* following is new featrue of CEA-861-F */
    HI_HDMI_COLOR_SPACE_YCBCR420,
    HI_HDMI_COLOR_SPACE_BUTT
} hi_hdmi_color_space;

typedef enum {
    HI_HDMI_BAR_INFO_NOT_VALID, /* Bar Data not valid */
    HI_HDMI_BAR_INFO_V,         /* Vertical bar data valid */
    HI_HDMI_BAR_INFO_H,         /* Horizontal bar data valid */
    HI_HDMI_BAR_INFO_VH,        /* Horizontal and Vertical bar data valid */
    HI_HDMI_BAR_INFO_BUTT
} hi_hdmi_bar_info;

typedef enum {
    HI_HDMI_SCAN_INFO_NO_DATA,      /* No Scan information */
    HI_HDMI_SCAN_INFO_OVERSCANNED,  /* Scan information, Overscanned (for television) */
    HI_HDMI_SCAN_INFO_UNDERSCANNED, /* Scan information, Underscanned (for computer) */
    HI_HDMI_SCAN_INFO_BUTT
} hi_hdmi_scan_info;

typedef enum {
    HI_HDMI_COMMON_COLORIMETRY_NO_DATA, /* Colorimetry No Data option */
    HI_HDMI_COMMON_COLORIMETRY_ITU601,  /* Colorimetry ITU601 option */
    HI_HDMI_COMMON_COLORIMETRY_ITU709,  /* Colorimetry ITU709 option */
    HI_HDMI_COMMON_COLORIMETRY_BUTT     /* Colorimetry extended option */
} hi_hdmi_colorimetry;

typedef enum {
    HI_HDMI_COMMON_COLORIMETRY_XVYCC_601,               /* Colorimetry xvYCC601 extended option */
    HI_HDMI_COMMON_COLORIMETRY_XVYCC_709,               /* Colorimetry xvYCC709 extended option */
    HI_HDMI_COMMON_COLORIMETRY_S_YCC_601,               /* Colorimetry S YCC 601 extended option */
    HI_HDMI_COMMON_COLORIMETRY_ADOBE_YCC_601,           /* Colorimetry ADOBE YCC 601 extended option */
    HI_HDMI_COMMON_COLORIMETRY_ADOBE_RGB,               /* Colorimetry ADOBE RGB extended option */
    HI_HDMI_COMMON_COLORIMETRY_2020_CONST_LUMINOUS,     /* Colorimetry ITU2020 extended option */
    HI_HDMI_COMMON_COLORIMETRY_2020_NON_CONST_LUMINOUS, /* Colorimetry ITU2020 extended option */
    HI_HDMI_COMMON_COLORIMETRY_EXT_BUTT
} hi_hdmi_ex_colorimetry;

typedef enum {
    HI_HDMI_PIC_ASPECT_RATIO_NO_DATA,
    HI_HDMI_PIC_ASPECT_RATIO_4TO3,
    HI_HDMI_PIC_ASPECT_RATIO_16TO9,
    HI_HDMI_PIC_ASPECT_RATIO_64TO27,
    HI_HDMI_PIC_ASPECT_RATIO_256TO135,
    HI_HDMI_PIC_ASPECT_RATIO_BUTT
} hi_pic_aspect_ratio;

typedef enum {
    HI_HDMI_ACTIVE_ASPECT_RATIO_16TO9_TOP = 2,
    HI_HDMI_ACTIVE_ASPECT_RATIO_14TO9_TOP,
    HI_HDMI_ACTIVE_ASPECT_RATIO_16TO9_BOX_CENTER,
    HI_HDMI_ACTIVE_ASPECT_RATIO_SAME_PIC = 8,
    HI_HDMI_ACTIVE_ASPECT_RATIO_4TO3_CENTER,
    HI_HDMI_ACTIVE_ASPECT_RATIO_16TO9_CENTER,
    HI_HDMI_ACTIVE_ASPECT_RATIO_14TO9_CENTER,
    HI_HDMI_ACTIVE_ASPECT_RATIO_4TO3_14_9 = 13,
    HI_HDMI_ACTIVE_ASPECT_RATIO_16TO9_14_9,
    HI_HDMI_ACTIVE_ASPECT_RATIO_16TO9_4_3,
    HI_HDMI_ACTIVE_ASPECT_RATIO_BUTT
} hi_hdmi_active_aspect_ratio;

typedef enum {
    HI_HDMI_PIC_NON_UNIFORM_SCALING, /* No Known, non-uniform picture scaling */
    HI_HDMI_PIC_SCALING_H,           /* Picture has been scaled horizontally */
    HI_HDMI_PIC_SCALING_V,           /* Picture has been scaled Vertically */
    HI_HDMI_PIC_SCALING_HV,          /* Picture has been scaled horizontally and Vertically */
    HI_HDMI_PIC_SCALING_BUTT
} hi_hdmi_pic_scaline;

typedef enum {
    HI_HDMI_RGB_QUANT_DEFAULT_RANGE, /* Default range, it depends on the video format */
    HI_HDMI_RGB_QUANT_LIMITED_RANGE, /* Limited quantization range of 220 levels when receiving a CE video format */
    HI_HDMI_RGB_QUANT_FULL_RANGE,    /* Full quantization range of 256 levels when receiving an IT video format */
    HI_HDMI_RGB_QUANT_FULL_BUTT
} hi_hdmi_rgb_quant_range;

typedef enum {
    HI_HDMI_PIXEL_REPET_NO,
    HI_HDMI_PIXEL_REPET_2_TIMES,
    HI_HDMI_PIXEL_REPET_3_TIMES,
    HI_HDMI_PIXEL_REPET_4_TIMES,
    HI_HDMI_PIXEL_REPET_5_TIMES,
    HI_HDMI_PIXEL_REPET_6_TIMES,
    HI_HDMI_PIXEL_REPET_7_TIMES,
    HI_HDMI_PIXEL_REPET_8_TIMES,
    HI_HDMI_PIXEL_REPET_9_TIMES,
    HI_HDMI_PIXEL_REPET_10_TIMES,
    HI_HDMI_PIXEL_REPET_BUTT
} hi_hdmi_pixel_repetition;

typedef enum {
    HI_HDMI_CONTNET_GRAPHIC,
    HI_HDMI_CONTNET_PHOTO,
    HI_HDMI_CONTNET_CINEMA,
    HI_HDMI_CONTNET_GAME,
    HI_HDMI_CONTNET_BUTT
} hi_hdmi_content_type;

typedef enum {
    HI_HDMI_YCC_QUANT_LIMITED_RANGE, /* Limited quantization range of 220 levels when receiving a CE video format */
    HI_HDMI_YCC_QUANT_FULL_RANGE,    /* Full quantization range of 256 levels when receiving an IT video format */
    HI_HDMI_YCC_QUANT_BUTT
} hi_hdmi_ycc_quant_range;

typedef struct {
    hi_hdmi_video_format timing_mode;
    hi_hdmi_color_space color_space;
    hi_bool active_info_present;
    hi_hdmi_bar_info bar_info;
    hi_hdmi_scan_info scan_info;
    hi_hdmi_colorimetry colorimetry;
    hi_hdmi_ex_colorimetry ex_colorimetry;
    hi_pic_aspect_ratio aspect_ratio;
    hi_hdmi_active_aspect_ratio active_aspect_ratio;
    hi_hdmi_pic_scaline pic_scaling;
    hi_hdmi_rgb_quant_range rgb_quant;
    hi_bool is_it_content;
    hi_hdmi_pixel_repetition pixel_repetition;
    hi_hdmi_content_type content_type;
    hi_hdmi_ycc_quant_range ycc_quant;
    hi_u16 line_n_end_of_top_bar;
    hi_u16 line_n_start_of_bot_bar;
    hi_u16 pixel_n_end_of_left_bar;
    hi_u16 pixel_n_start_of_right_bar;
} hi_hdmi_avi_infoframe;

typedef enum {
    HI_HDMI_AUDIO_CHN_CNT_STREAM,
    HI_HDMI_AUDIO_CHN_CNT_2,
    HI_HDMI_AUDIO_CHN_CNT_3,
    HI_HDMI_AUDIO_CHN_CNT_4,
    HI_HDMI_AUDIO_CHN_CNT_5,
    HI_HDMI_AUDIO_CHN_CNT_6,
    HI_HDMI_AUDIO_CHN_CNT_7,
    HI_HDMI_AUDIO_CHN_CNT_8,
    HI_HDMI_AUDIO_CHN_CNT_BUTT
} hi_hdmi_audio_chn_cnt;

typedef enum {
    HI_HDMI_AUDIO_CODING_REFER_STREAM_HEAD,
    HI_HDMI_AUDIO_CODING_PCM,
    HI_HDMI_AUDIO_CODING_AC3,
    HI_HDMI_AUDIO_CODING_MPEG1,
    HI_HDMI_AUDIO_CODING_MP3,
    HI_HDMI_AUDIO_CODING_MPEG2,
    HI_HDMI_AUDIO_CODING_AACLC,
    HI_HDMI_AUDIO_CODING_DTS,
    HI_HDMI_AUDIO_CODING_ATRAC,
    HI_HDMI_AUDIO_CODIND_ONE_BIT_AUDIO,
    HI_HDMI_AUDIO_CODING_ENAHNCED_AC3,
    HI_HDMI_AUDIO_CODING_DTS_HD,
    HI_HDMI_AUDIO_CODING_MAT,
    HI_HDMI_AUDIO_CODING_DST,
    HI_HDMI_AUDIO_CODING_WMA_PRO,
    HI_HDMI_AUDIO_CODING_BUTT
} hi_hdmi_coding_type;

typedef enum {
    HI_HDMI_AUDIO_SAMPLE_SIZE_STREAM,
    HI_HDMI_AUDIO_SAMPLE_SIZE_16,
    HI_HDMI_AUDIO_SAMPLE_SIZE_20,
    HI_HDMI_AUDIO_SAMPLE_SIZE_24,
    HI_HDMI_AUDIO_SAMPLE_SIZE_BUTT
} hi_hdmi_audio_sample_size;

typedef enum {
    HI_HDMI_AUDIO_SAMPLE_FREQ_STREAM,
    HI_HDMI_AUDIO_SAMPLE_FREQ_32000,
    HI_HDMI_AUDIO_SAMPLE_FREQ_44100,
    HI_HDMI_AUDIO_SAMPLE_FREQ_48000,
    HI_HDMI_AUDIO_SAMPLE_FREQ_88200,
    HI_HDMI_AUDIO_SAMPLE_FREQ_96000,
    HI_HDMI_AUDIO_SAMPLE_FREQ_176400,
    HI_HDMI_AUDIO_SAMPLE_FREQ_192000,
    HI_HDMI_AUDIO_SAMPLE_FREQ_BUTT
} hi_hdmi_audio_sample_freq;

typedef enum {
    HI_HDMI_LEVEL_SHIFT_VAL_0_DB,
    HI_HDMI_LEVEL_SHIFT_VAL_1_DB,
    HI_HDMI_LEVEL_SHIFT_VAL_2_DB,
    HI_HDMI_LEVEL_SHIFT_VAL_3_DB,
    HI_HDMI_LEVEL_SHIFT_VAL_4_DB,
    HI_HDMI_LEVEL_SHIFT_VAL_5_DB,
    HI_HDMI_LEVEL_SHIFT_VAL_6_DB,
    HI_HDMI_LEVEL_SHIFT_VAL_7_DB,
    HI_HDMI_LEVEL_SHIFT_VAL_8_DB,
    HI_HDMI_LEVEL_SHIFT_VAL_9_DB,
    HI_HDMI_LEVEL_SHIFT_VAL_10_DB,
    HI_HDMI_LEVEL_SHIFT_VAL_11_DB,
    HI_HDMI_LEVEL_SHIFT_VAL_12_DB,
    HI_HDMI_LEVEL_SHIFT_VAL_13_DB,
    HI_HDMI_LEVEL_SHIFT_VAL_14_DB,
    HI_HDMI_LEVEL_SHIFT_VAL_15_DB,
    HI_HDMI_LEVEL_SHIFT_VAL_BUTT
} hi_hdmi_level_shift_val;

typedef enum {
    HI_HDMI_LFE_PLAYBACK_NO,
    HI_HDMI_LFE_PLAYBACK_0_DB,
    HI_HDMI_LFE_PLAYBACK_10_DB,
    HI_HDMI_LFE_PLAYBACK_BUTT
} hi_hdmi_lfe_playback_level;


typedef struct {
    hi_hdmi_audio_chn_cnt chn_cnt;
    hi_hdmi_coding_type coding_type;
    hi_hdmi_audio_sample_size sample_size;
    hi_hdmi_audio_sample_freq sampling_freq;
    hi_u8 chn_alloc; /* Channel/Speaker Allocation.Range [0,255] */
    hi_hdmi_level_shift_val level_shift;
    hi_hdmi_lfe_playback_level lfe_playback_level;
    hi_bool down_mix_inhibit;
} hi_hdmi_audio_infoframe;

#define OT_HDMI_VENDOR_USER_DATA_MAX_LEN 22
typedef struct {
    hi_u8 data_len;
    hi_u8 user_data[OT_HDMI_VENDOR_USER_DATA_MAX_LEN];
} hi_hdmi_vendorspec_infoframe;

typedef union {
    hi_hdmi_avi_infoframe avi_infoframe;                /* AUTO:ot_hdmi_infoframe_type:OT_INFOFRAME_TYPE_AVI; */
    hi_hdmi_audio_infoframe audio_infoframe;            /* AUTO:ot_hdmi_infoframe_type:OT_INFOFRAME_TYPE_AUDIO; */
    hi_hdmi_vendorspec_infoframe vendor_spec_infoframe; /* AUTO:ot_hdmi_infoframe_type:OT_INFOFRAME_TYPE_VENDORSPEC; */
} hi_hdmi_infoframe_unit;

typedef struct {
    hi_hdmi_infoframe_type infoframe_type; /* InfoFrame type */
    hi_hdmi_infoframe_unit infoframe_unit; /* InfoFrame date */
} hi_hdmi_infoframe;

typedef struct {
    hi_bool is_connected; /* Whether the devices are connected. */
    /* Whether the HDMI is supported by the device. If the HDMI is not supported by the device, the device is DVI. */
    hi_bool support_hdmi;
    /* Whether to support HDMI2.0. */
    hi_bool support_hdmi_2_0;
} hi_hdmi_sink_capability;

typedef enum {
    HI_HDMI_EVENT_HOTPLUG = 0x10, /* HDMI hot-plug event */
    HI_HDMI_EVENT_NO_PLUG,        /* HDMI cable disconnection event */
    HI_HDMI_EVENT_EDID_FAIL,      /* reserve */
    HI_HDMI_EVENT_BUTT
} hi_hdmi_event_type;

typedef void (*hi_hdmi_callback)(hi_hdmi_event_type event, void *private_data);

typedef struct {
    hi_hdmi_callback hdmi_event_callback; /* Event handling callback function */
    void *private_data;                /* Private data of the callback functions and parameters */
} hi_hdmi_callback_func;

hi_s32 hi_mpi_hdmi_init(void);
hi_s32 hi_mpi_hdmi_deinit(void);
hi_s32 hi_mpi_hdmi_open(hi_hdmi_id hdmi);
hi_s32 hi_mpi_hdmi_close(hi_hdmi_id hdmi);
hi_s32 hi_mpi_hdmi_set_attr(hi_hdmi_id hdmi, const hi_hdmi_attr *attr);
hi_s32 hi_mpi_hdmi_get_attr(hi_hdmi_id hdmi, hi_hdmi_attr *attr);
hi_s32 hi_mpi_hdmi_start(hi_hdmi_id hdmi);
hi_s32 hi_mpi_hdmi_stop(hi_hdmi_id hdmi);
hi_s32 hi_mpi_hdmi_set_infoframe(hi_hdmi_id hdmi, const hi_hdmi_infoframe *infoframe);
hi_s32 hi_mpi_hdmi_get_sink_capability(hi_hdmi_id hdmi, hi_hdmi_sink_capability *capability);
hi_s32 hi_mpi_hdmi_register_callback(hi_hdmi_id hdmi, const hi_hdmi_callback_func *callback_func);

#ifdef __cplusplus
}
#endif
#endif /* __HI_MPI_HDMI_H__ */
