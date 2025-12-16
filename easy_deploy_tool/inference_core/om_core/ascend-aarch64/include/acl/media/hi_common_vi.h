/*
 * Copyright (C) Hisilicon Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Vi mpi interface
 * Author: Hisilicon multimedia software group
 * Create: 2022/12/07
 */

#ifndef HI_COMMON_VI_H
#define HI_COMMON_VI_H

#include "hi_media_type.h"
#include "hi_media_common.h"
#include "hi_common_gdc.h"
#include "hi_common_dis.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* __cplusplus */

typedef hi_s32 hi_sensor_id;
typedef hi_s32 hi_vi_dev;
typedef hi_s32 hi_vi_pipe;
typedef hi_s32 hi_vi_chn;
typedef hi_s32 hi_vi_stitch_grp;
typedef hi_s32 hi_vi_mcf_grp;
typedef hi_s32 hi_isp_dev;
typedef hi_s32 hi_slave_dev;

#define HI_VI_MAX_DEV_NUM                  4
#define HI_VI_MAX_PHY_PIPE_NUM             4
#define HI_VI_MAX_VIR_PIPE_NUM             8
#define HI_VI_MAX_PIPE_NUM                 (HI_VI_MAX_PHY_PIPE_NUM + HI_VI_MAX_VIR_PIPE_NUM)

typedef enum hiEN_VI_ERR_CODE_E {
    ERR_VI_FAILED_NOTENABLE = 64,            /* device or channel not enable */
    ERR_VI_FAILED_NOTDISABLE = 65,           /* device not disable */
    ERR_VI_CFG_TIMEOUT = 67,                 /* config timeout */
    ERR_VI_FAILED_NOTBIND = 71,              /* device or channel not bind */
    ERR_VI_FAILED_BINDED = 72,               /* device or channel not unbind */
} hi_vi_err_code;

#define HI_ERR_VI_ILLEGAL_PARAM       HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, HI_ERR_ILLEGAL_PARAM)
#define HI_ERR_VI_INVALID_DEV_ID      HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, HI_ERR_INVALID_DEV_ID)
#define HI_ERR_VI_INVALID_PIPE_ID     HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, HI_ERR_INVALID_PIPE_ID)
#define HI_ERR_VI_INVALID_CHN_ID      HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, HI_ERR_INVALID_CHN_ID)
#define HI_ERR_VI_NULL_PTR            HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, HI_ERR_NULL_PTR)
#define HI_ERR_VI_NOT_CFG             HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, HI_ERR_NOT_CFG)
#define HI_ERR_VI_SYS_NOT_READY       HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, HI_ERR_NOT_READY)
#define HI_ERR_VI_BUF_EMPTY           HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, HI_ERR_BUF_EMPTY)
#define HI_ERR_VI_BUF_FULL            HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, HI_ERR_BUF_FULL)
#define HI_ERR_VI_NO_MEM              HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, HI_ERR_NO_MEM)
#define HI_ERR_VI_NOT_SUPPORT         HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, HI_ERR_NOT_SUPPORT)
#define HI_ERR_VI_BUSY                HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, HI_ERR_BUSY)
#define HI_ERR_VI_NOT_PERM            HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, HI_ERR_NOT_PERM)
#define HI_ERR_VI_PIPE_EXIST          HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, HI_ERR_EXIST)
#define HI_ERR_VI_PIPE_UNEXIST        HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, HI_ERR_UNEXIST)

#define HI_ERR_VI_NOT_ENABLE          HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, ERR_VI_FAILED_NOTENABLE)
#define HI_ERR_VI_NOT_DISABLE         HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, ERR_VI_FAILED_NOTDISABLE)
#define HI_ERR_VI_TIME_OUT            HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, ERR_VI_CFG_TIMEOUT)
#define HI_ERR_VI_NOT_BIND            HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, ERR_VI_FAILED_NOTBIND)
#define HI_ERR_VI_BINDED              HI_DEFINE_ERR(HI_ID_VI, HI_ERR_LEVEL_ERROR, ERR_VI_FAILED_BINDED)

typedef enum hiWDR_MODE_E {
    HI_WDR_MODE_NONE = 0,
    HI_WDR_MODE_BUILT_IN,
    HI_WDR_MODE_QUDRA,

    HI_WDR_MODE_2To1_LINE,
    HI_WDR_MODE_2To1_FRAME,

    HI_WDR_MODE_3To1_LINE,
    HI_WDR_MODE_3To1_FRAME,

    HI_WDR_MODE_4To1_LINE,
    HI_WDR_MODE_4To1_FRAME,

    HI_WDR_MODE_BUTT,
} hi_wdr_mode;

typedef enum hiDATA_RATE_E {
    HI_DATA_RATE_X1 = 0,         /* RW; output 1 pixel per clock */
    HI_DATA_RATE_X2 = 1,         /* RW; output 2 pixel per clock */

    HI_DATA_RATE_BUTT
} hi_data_rate;

/* Information of pipe binded to device */
typedef struct {
    hi_u32  num;                                     /* RW;Range [1,VI_MAX_PHY_PIPE_NUM] */
    hi_vi_pipe pipe_id[HI_VI_MAX_PHY_PIPE_NUM];                /* RW;Array of pipe ID */
} hi_vi_dev_bind_pipe;

typedef enum hiVI_PIPE_BYPASS_MODE_E {
    HI_VI_PIPE_BYPASS_NONE,
    HI_VI_PIPE_BYPASS_FE,
    HI_VI_PIPE_BYPASS_BE,

    HI_VI_PIPE_BYPASS_BUTT
} hi_vi_pipe_bypass_mode;

/* The attributes of pipe */
typedef struct {
    hi_vi_pipe_bypass_mode pipe_bypass_mode;
    hi_bool               isp_bypass;             /* RW;Range:[0, 1];ISP bypass enable */
    hi_size               size;                    /* RW;size */
    hi_pixel_format       pixel_format;               /* RW;Pixel format */
    hi_compress_mode      compress_mode;         /* RW;Range:[0, 4];Compress mode. */
    hi_data_bit_width     bit_width;             /* RW;Range:[0, 4];Bit width */
    hi_frame_rate_ctrl    frame_rate_ctrl;            /* RW;Frame rate */
    hi_u32                depth;                 /* Range:(0, 16]; default is 3 */
    hi_u32                reserved[10]; /* RW;3DNR X param version 1 */
} hi_vi_pipe_attr;

typedef enum hiVI_INTF_MODE_E {
    HI_VI_MODE_MIPI = 4,                   /* MIPI RAW mode */
    HI_VI_MODE_MIPI_YUV420_NORMAL,     /* MIPI YUV420 normal mode */
    HI_VI_MODE_MIPI_YUV420_LEGACY,     /* MIPI YUV420 legacy mode */
    HI_VI_MODE_MIPI_YUV422,            /* MIPI YUV422 mode */
    HI_VI_MODE_LVDS,                   /* LVDS mode */
    HI_VI_MODE_HISPI,                  /* HiSPi mode */
    HI_VI_MODE_SLVS,                   /* SLVS mode */

    HI_VI_MODE_BUTT
} hi_vi_intf_mode;

typedef enum hiVI_SCAN_MODE_E {
    HI_VI_SCAN_PROGRESSIVE = 0,   /* progressive mode */
    HI_VI_SCAN_INTERLACED,        /* interlaced mode */
    HI_VI_SCAN_BUTT
} hi_vi_scan_mode;

typedef enum hiVI_YUV_DATA_SEQ_E {
    HI_VI_DATA_SEQ_VUVU = 0,   /* The input sequence of the second component(only contains u and v) in BT.
                            1120 mode is VUVU */
    HI_VI_DATA_SEQ_UVUV,       /* The input sequence of the second component(only contains u and v) in BT.
                            1120 mode is UVUV */

    HI_VI_DATA_SEQ_UYVY,       /* The input sequence of YUV is UYVY */
    HI_VI_DATA_SEQ_VYUY,       /* The input sequence of YUV is VYUY */
    HI_VI_DATA_SEQ_YUYV,       /* The input sequence of YUV is YUYV */
    HI_VI_DATA_SEQ_YVYU,       /* The input sequence of YUV is YVYU */

    HI_VI_DATA_SEQ_BUTT
} hi_vi_data_seq;

/* Input data type */
typedef enum hiVI_DATA_TYPE_E {
    HI_VI_DATA_TYPE_RAW = 0,
    HI_VI_DATA_TYPE_YUV,

    HI_VI_DATA_TYPE_BUTT
} hi_vi_data_type;

/* Attribute of wdr */
typedef struct {
    hi_wdr_mode  wdr_mode;        /* RW; WDR mode. */
    hi_u32      cache_line;       /* RW; WDR cache line. */
} hi_vi_wdr_attr;

typedef struct {
    hi_bool         enable;              /* RW;whether dump is enable */
    hi_u32          depth;               /* RW;range [0,8];depth */
} hi_vi_frame_dump_attr;

typedef enum hiVI_PIPE_FRAME_SOURCE_E {
    HI_VI_PIPE_FRAME_SOURCE_FE = 0, /* RW;Source from dev */
    HI_VI_PIPE_FRAME_SOURCE_USER, /* RW;User send to BE */

    HI_VI_PIPE_FRAME_SOURCE_BUTT
} hi_vi_pipe_frame_source;

/* the attributes of channel */
typedef struct {
    hi_size              size;             /* RW;channel out put size */
    hi_pixel_format      pixel_format;     /* RW;pixel format */
    hi_dynamic_range     dynamic_range;    /* RW;dynamic range */
    hi_video_format      video_format;     /* RW;video format */
    hi_compress_mode     compress_mode;    /* RW;256B segment compress or no compress. */
    hi_u32               depth;            /* RW;range [0,8];depth */
    hi_frame_rate_ctrl   frame_rate_ctrl;  /* RW;frame rate */
    hi_u32               reserved[10];
} hi_vi_chn_attr;

/* The attributes of a VI device */
typedef struct {
    hi_vi_intf_mode      intf_mode;                     /* RW;Interface mode */
    hi_vi_scan_mode      scan_mode;                     /* RW;Input scanning mode (progressive or interlaced) */
    /* The below members must be configured in BT.601 mode or DC mode and are invalid in other modes */
    hi_vi_data_seq       data_seq;                      /* RW;Input data sequence (only the YUV format is supported) */

    hi_vi_data_type      data_type;                     /* RW;RGB: CSC-709 or CSC-601, PT YUV444 disable; YUV: default
                                                            yuv CSC coef PT YUV444 enable. */
    hi_size              in_size;                       /* RW;Input size */

    hi_vi_wdr_attr       wdr_attr;                      /* RW;Attribute of WDR */

    hi_data_rate         data_rate;                     /* RW;Data rate of device */

    hi_u32               reserved[30];
} hi_vi_dev_attr;

/* the attributes of LDC */
typedef struct {
    hi_bool enable; /* RW;range [0,1];whether LDC is enbale */
    hi_ldc_version ldc_version;
    union {
        hi_ldc_v1_attr ldc_v1_attr;
        hi_ldc_v2_attr ldc_v2_attr;
    };
} hi_vi_ldc_attr;

typedef struct {
    hi_bool enable;          /* RW; Low delay enable. */
    hi_u32 line_cnt;        /* RW; Range: [32, 16384]; Low delay shoreline. */
} hi_vi_low_delay_info;

typedef struct {
    hi_u32      iso;                     /* ISP internal ISO : again*dgain*is_pgain */
    hi_u32      exposure_time;           /* exposure time (reciprocal of shutter speed),unit is us */
    hi_u32      isp_dgain;
    hi_u32      again;
    hi_u32      dgain;
    hi_u32      ratio[3];
    hi_u32      isp_nr_strength;
    hi_u32      f_number;                /* the actual F-number (F-stop) of lens when the image was taken */
    hi_u32      sensor_id;               /* which sensor is used */
    hi_u32      sensor_mode;
    hi_u32      hmax_times;              /* sensor hmax_times,unit is ns */
    hi_u32      vmax;                    /* sensor vmax,unit is line */
    hi_u32      vc_num;                  /* when dump wdr frame, which is long or short  exposure frame. */
    hi_u64      fsync_timestamp;         /* sensorhub timestamp */
    hi_u64      fs_timestamp;            /* vi cap start timestamp */
    hi_u64      fe_timestamp;            /* vi cap end timestamp */
    hi_u64      pe_timestamp;            /* vpss process end timestamp */
    hi_s64      motion_filter[HI_DIS_MOTION_FILTER_SIZE];      /* DIS motion filter */
    hi_u32      reserve[10];             /* reserve 40 bytes */
} hi_isp_frame_info;

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */

#endif /* __MPI_VI_H__ */
