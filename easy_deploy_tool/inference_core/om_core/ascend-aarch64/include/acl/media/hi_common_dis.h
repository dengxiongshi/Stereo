/*
 * Copyright (C) Hisilicon Technologies Co., Ltd. 2016-2019. All rights reserved.
 * Description: Common struct definition for DIS
 * Author: Hisilicon multimedia software group
 * Create: 2016-07-25
 */
#ifndef HI_COMMON_DIS_H
#define HI_COMMON_DIS_H

#include "hi_media_type.h"
#include "hi_media_common.h"
#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* __cplusplus */

#define HI_DIS_MOTION_FILTER_SIZE             9

/* failure caused by malloc buffer */
#define HI_ERR_DIS_NOBUF         HI_DEFINE_ERR(HI_ID_DIS, EN_ERR_LEVEL_ERROR, EN_ERR_NOBUF)
#define HI_ERR_DIS_BUF_EMPTY     HI_DEFINE_ERR(HI_ID_DIS, EN_ERR_LEVEL_ERROR, EN_ERR_BUF_EMPTY)
#define HI_ERR_DIS_NULL_PTR      HI_DEFINE_ERR(HI_ID_DIS, EN_ERR_LEVEL_ERROR, EN_ERR_NULL_PTR)
#define HI_ERR_DIS_ILLEGAL_PARAM HI_DEFINE_ERR(HI_ID_DIS, EN_ERR_LEVEL_ERROR, EN_ERR_ILLEGAL_PARAM)
#define HI_ERR_DIS_BUF_FULL      HI_DEFINE_ERR(HI_ID_DIS, EN_ERR_LEVEL_ERROR, EN_ERR_BUF_FULL)
#define HI_ERR_DIS_SYS_NOTREADY  HI_DEFINE_ERR(HI_ID_DIS, EN_ERR_LEVEL_ERROR, EN_ERR_SYS_NOTREADY)
#define HI_ERR_DIS_NOT_SUPPORT   HI_DEFINE_ERR(HI_ID_DIS, EN_ERR_LEVEL_ERROR, EN_ERR_NOT_SUPPORT)
#define HI_ERR_DIS_NOT_PERMITTED HI_DEFINE_ERR(HI_ID_DIS, EN_ERR_LEVEL_ERROR, EN_ERR_NOT_PERM)
#define HI_ERR_DIS_BUSY          HI_DEFINE_ERR(HI_ID_DIS, EN_ERR_LEVEL_ERROR, EN_ERR_BUSY)
#define HI_ERR_DIS_INVALID_CHNID HI_DEFINE_ERR(HI_ID_DIS, EN_ERR_LEVEL_ERROR, EN_ERR_INVALID_CHNID)
#define HI_ERR_DIS_CHN_UNEXIST   HI_DEFINE_ERR(HI_ID_DIS, EN_ERR_LEVEL_ERROR, EN_ERR_UNEXIST)

/* Different mode of DIS */
typedef enum {
    HI_DIS_MODE_4_DOF_GME = 0, /* Only use with GME in 4 dof  */
    HI_DIS_MODE_6_DOF_GME,     /* Only use with GME in 6 dof  */
    HI_DIS_MODE_GYRO,          /* Only use with gryo in 6 dof  */
    HI_DIS_MODE_DOF_BUTT
} hi_dis_mode;

/* The motion level of camera */
typedef enum {
    HI_DIS_MOTION_LEVEL_LOW = 0,   /* Low motion level */
    HI_DIS_MOTION_LEVEL_NORMAL,    /* Normal motion level */
    HI_DIS_MOTION_LEVEL_HIGH,      /* High motion level */
    HI_DIS_MOTION_LEVEL_BUTT
} hi_dis_motion_level;

/* Different product type used DIS */
typedef enum {
    HI_DIS_PDT_TYPE_IPC = 0,   /* IPC product type */
    HI_DIS_PDT_TYPE_DV,        /* DV product type */
    HI_DIS_PDT_TYPE_DRONE,     /* DRONE product type */
    HI_DIS_PDT_TYPE_BUTT
} hi_dis_pdt_type;

/* The Parameter of DIS */
typedef struct {
    /*
     * RW; [0,100],
     * 0: attenuate large motion most in advance,
     * 100: never attenuate large motion;
     * larger value -> better stability but more likely to crop to the border with large motion
     */
    hi_u32 large_motion_stable_coef;
    /*
     * RW; [0,100],
     * 0: never preserve the low frequency motion,
     * 100: keep all the low frequency motion;
     * small value -> better stability but more likely to crop to the border even with low level motion
     */
    hi_u32 low_freq_motion_preserve;
    /*
     * RW; [0,100],
     * 0: lowest cut frequency,
     * 100: highest cut frequency;
     * small value -> better stability but more likely to crop to the border even with large motion
     */
    hi_u32 low_freq_motion_freq;
} hi_dis_param;

/* The Attribute of DIS */
typedef struct {
    hi_bool enable;               /* RW; DIS enable */
    hi_bool gdc_bypass;           /* RW; gdc correction process , DIS = GME&GDC correction */
    hi_u32  moving_subject_level; /* RW; Range:[0,6]; Moving Subject level */
    hi_s32  rolling_shutter_coef; /* RW; Range:[0,1000]; Rolling shutter coefficients */
    hi_s32  timelag;              /* RW; Range:[-2000000,2000000]; Timestamp delay between Gyro and Frame PTS */
    hi_u32  view_angle;           /* Reserved */
    hi_u32  horizontal_limit;     /* RW; Range:[0,1000]; Parameter to limit horizontal drift by large foreground */
    hi_u32  vertical_limit;       /* RW; Range:[0,1000]; Parameter to limit vertical drift by large foreground */
    hi_bool still_crop;           /* RW; The stabilization not working,but the output image still be cropped */
    hi_u32  strength;             /* RW. The DIS strength for different light */
} hi_dis_attr;

/* The Config of DIS */
typedef struct {
    hi_dis_mode         mode;                  /* RW; DIS Mode */
    hi_dis_motion_level motion_level;          /* RW; DIS Motion level of the camera */
    hi_dis_pdt_type     pdt_type;              /* RW; DIS product type */
    hi_u32              buf_num;               /* RW; Range:[5,10]; Buf num for DIS */
    hi_u32              crop_ratio;            /* RW; Range:[50,98]; Crop ratio of output image */
    hi_u32              frame_rate;            /* RW; Range:(0, 60]; */
    hi_bool             camera_steady;         /* RW; The camera is steady or not */
    hi_bool             scale;                 /* RW; Scale output image or not */
} hi_dis_config;

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */

#endif /* __HI_COMM_DIS_H__ */
