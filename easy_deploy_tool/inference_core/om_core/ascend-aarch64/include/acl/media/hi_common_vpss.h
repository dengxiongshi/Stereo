/*
 * Copyright (C) Hisilicon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: Vpss mpi interface
 * Author: Hisilicon multimedia software group
 * Create: 2022/12/27
 */

#ifndef HI_COMMON_VPSS_H
#define HI_COMMON_VPSS_H

#include "hi_media_type.h"
#include "hi_media_common.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* __cplusplus */

typedef hi_s32 hi_vpss_grp;
typedef hi_s32 hi_vpss_chn;

#define HI_VPSS_MAX_GRP_NUM                268U
#define HI_VPSS_VIPE_START_GRP_ID          256U
#define HI_VPSS_MAX_CHN_NUM                2U

#define HI_ERR_VPSS_NULL_PTR         HI_DEFINE_ERR(HI_ID_VPSS, HI_ERR_LEVEL_ERROR, HI_ERR_NULL_PTR)
#define HI_ERR_VPSS_NOT_READY        HI_DEFINE_ERR(HI_ID_VPSS, HI_ERR_LEVEL_ERROR, HI_ERR_NOT_READY)
#define HI_ERR_VPSS_INVALID_DEV_ID   HI_DEFINE_ERR(HI_ID_VPSS, HI_ERR_LEVEL_ERROR, HI_ERR_INVALID_DEV_ID)
#define HI_ERR_VPSS_INVALID_CHN_ID   HI_DEFINE_ERR(HI_ID_VPSS, HI_ERR_LEVEL_ERROR, HI_ERR_INVALID_CHN_ID)
#define HI_ERR_VPSS_EXIST            HI_DEFINE_ERR(HI_ID_VPSS, HI_ERR_LEVEL_ERROR, HI_ERR_EXIST)
#define HI_ERR_VPSS_UNEXIST          HI_DEFINE_ERR(HI_ID_VPSS, HI_ERR_LEVEL_ERROR, HI_ERR_UNEXIST)
#define HI_ERR_VPSS_NOT_SUPPORT      HI_DEFINE_ERR(HI_ID_VPSS, HI_ERR_LEVEL_ERROR, HI_ERR_NOT_SUPPORT)
#define HI_ERR_VPSS_NOT_PERM         HI_DEFINE_ERR(HI_ID_VPSS, HI_ERR_LEVEL_ERROR, HI_ERR_NOT_PERM)
#define HI_ERR_VPSS_NO_MEM           HI_DEFINE_ERR(HI_ID_VPSS, HI_ERR_LEVEL_ERROR, HI_ERR_NO_MEM)
#define HI_ERR_VPSS_NOBUF            HI_DEFINE_ERR(HI_ID_VPSS, HI_ERR_LEVEL_ERROR, HI_ERR_NO_BUF)
#define HI_ERR_VPSS_SIZE_NOT_ENOUGH  HI_DEFINE_ERR(HI_ID_VPSS, HI_ERR_LEVEL_ERROR, HI_ERR_SIZE_NOT_ENOUGH)
#define HI_ERR_VPSS_ILLEGAL_PARAM    HI_DEFINE_ERR(HI_ID_VPSS, HI_ERR_LEVEL_ERROR, HI_ERR_ILLEGAL_PARAM)
#define HI_ERR_VPSS_BUSY             HI_DEFINE_ERR(HI_ID_VPSS, HI_ERR_LEVEL_ERROR, HI_ERR_BUSY)
#define HI_ERR_VPSS_BUF_EMPTY        HI_DEFINE_ERR(HI_ID_VPSS, HI_ERR_LEVEL_ERROR, HI_ERR_BUF_EMPTY)

typedef enum {
    HI_VPSS_NR_TYPE_VIDEO_NORM     = 0,
    HI_VPSS_NR_TYPE_BUTT
} hi_vpss_nr_type;

typedef enum {
    HI_VPSS_NR_MOTION_MODE_NORM    = 0,        /* normal */
    HI_VPSS_NR_MOTION_BUTT
} hi_vpss_nr_motion_mode;

typedef enum {
    HI_VPSS_CHN_MODE_AUTO  = 0,       /* Auto mode. */
    HI_VPSS_CHN_MODE_USER  = 1,       /* User mode. */
    HI_VPSS_CHN_MODE_BUTT
} hi_vpss_chn_mode;

typedef struct {
    hi_vpss_nr_type      nr_type;
    hi_compress_mode     compress_mode;   /* RW; reference frame compress mode */
    hi_vpss_nr_motion_mode    nr_motion_mode;   /* RW; NR motion compensate mode. */
} hi_vpss_nr_attr;

typedef struct {
    hi_u32                     max_width;
    hi_u32                     max_height;
    hi_pixel_format             pixel_format;     /* RW; pixel format of source image. */
    hi_dynamic_range            dynamic_range;    /* RW; dynamic_range of source image. */
    hi_frame_rate_ctrl          frame_rate;       /* grp frame rate control. */
    hi_bool                    nr_en;             /* RW;range: [0, 1];  NR enable. */
    hi_vpss_nr_attr             nr_attr;          /* RW; NR attr. */
    hi_u32                     reserved[20];
} hi_vpss_grp_attr;

typedef struct {
    hi_bool             mirror_en;            /* mirror enable */
    hi_bool             flip_en;              /* flip enable */
    hi_vpss_chn_mode    chn_mode;          /* RW; vpss channel's work mode. */
    hi_u32              width;
    hi_u32              height;
    hi_video_format     video_format;      /* RW; video format of target image. */
    hi_pixel_format     pixel_format;      /* RW; pixel format of target image. */
    hi_dynamic_range    dynamic_range;     /* RW; dynamic_range of target image. */
    hi_compress_mode    compress_mode;     /* RW; compression mode of the output. */
    hi_frame_rate_ctrl  frame_rate;        /* frame rate control info */
    hi_u32              depth;             /* RW; range: [0, 8]; chn buffer total. */
    hi_aspect_ratio     aspect_ratio;      /* aspect ratio info. */
    hi_u32              reserved[10];
} hi_vpss_chn_attr;

typedef struct {
    hi_u8  ies0, ies1, ies2, ies3;
    hi_u16  iedz : 10, ie_en : 1, reserved : 5;
} hi_v200_vpss_iey;

typedef struct {
    hi_u8  spn6 : 3, sfr  : 5;                                      /* spn6, sbn6:  [0, 5]; */
    hi_u8  sbn6 : 3, pbr6 : 5;                                    /* sfr: [0,31];  pbr6: [0,15]; */
    hi_u16  srt0 : 5, srt1 : 5, j_mode : 3, de_idx : 3;
    hi_u8  sfr6[4], sbr6[2], de_rate;
    hi_u8  sfs1,  sft1,  sbr1;
    hi_u8  sfs2,  sft2,  sbr2;
    hi_u8  sfs4,  sft4,  sbr4;
    hi_u16  sth1 : 9,  sfn1 : 3, sfn0  : 3, nr_y_en   : 1;
    hi_u16  sth_d1 : 9, reserved : 7;
    hi_u16  sth2 : 9,  sfn2 : 3, k_mode : 3, reserved_1   : 1;
    hi_u16  sth_d2 : 9, reserved_2 : 7;
    hi_u16  sbs_k[32], sds_k[32];
} hi_v200_vpss_sfy;

typedef struct {
    hi_u16  tfs0 : 4,   tdz0 : 10,  tdx0    : 2;
    hi_u16  tfs1 : 4,   tdz1 : 10,  tdx1    : 2;
    hi_u16  sdz0 : 10,  str0 : 5,   dz_mode0 : 1;
    hi_u16  sdz1 : 10,  str1 : 5,   dz_mode1 : 1;
    hi_u8  tss0 : 4,   tsi0 : 4,  tfr0[6];
    hi_u8  tss1 : 4,   tsi1 : 4,  tfr1[6];
    hi_u8  tfrs : 4,   ted  : 2,   ref    : 1,  reserved : 1;
} hi_v200_vpss_tfy;

typedef struct {
    hi_u16  madz0   : 9,   mai00 : 2,  mai01  : 2, mai02 : 2, reserved : 1;
    hi_u16  madz1   : 9,   mai10 : 2,  mai11  : 2, mai12 : 2, reserved_1 : 1;
    hi_u8  mabr0, mabr1;
    hi_u16  math0   : 10,  mate0 : 4,  matw   : 2;
    hi_u16  math_d0  : 10,  reserved_2 : 6;
    hi_u16  math1   : 10,  reserved_3 : 6;
    hi_u16  math_d1  : 10,  reserved_4 : 6;
    hi_u8  masw    :  4,  mate1 : 4;
    hi_u8  mabw0   :  4,  mabw1 : 4;
    hi_u16  adv_math : 1,   adv_th : 12, reserved_5  : 3;
} hi_v200_vpss_mdy;

typedef struct {
    hi_u8  sfc, tfc : 6, reserved : 2;
    hi_u8  trc, tpc : 6, reserved_1 : 2;
    hi_u8 mode : 1, reserved_2 : 7;
    hi_u8  presfc : 6, reserved_3 : 2;
} hi_v200_vpss_nr_c;

typedef struct {
    hi_v200_vpss_iey  iey[5];
    hi_v200_vpss_sfy  sfy[5];
    hi_v200_vpss_mdy  mdy[2];
    hi_v200_vpss_tfy  tfy[3];
    hi_v200_vpss_nr_c  nr_c;
} hi_vpss_nrx_v3;

typedef struct {
    hi_vpss_nrx_v3 nrx_param;
} hi_vpss_nrx_param_manual_v3;

typedef struct {
    hi_u32 param_num;
    hi_u32 *iso;
    hi_vpss_nrx_v3 *nrx_param;
} hi_vpss_nrx_param_auto_v3;

typedef struct {
    hi_op_mode                  opt_mode;           /* RW;adaptive NR */
    hi_vpss_nrx_param_manual_v3 nrx_manual;         /* RW;NRX V3 param for manual video */
    hi_vpss_nrx_param_auto_v3   nrx_auto;           /* RW;NRX V3 param for auto video */
} hi_vpss_nrx_param_v3;

typedef enum {
    HI_VPSS_NR_V3 = 3,
    HI_VPSS_NR_BUTT
} hi_vpss_nr_version;

typedef struct {
    hi_vpss_nr_version nr_version;
    union {
        hi_vpss_nrx_param_v3 nrx_param_v3;
    };
} hi_vpss_grp_nrx_param;

typedef struct {
    hi_vpss_grp_nrx_param   nrx_param;
    hi_u32                  reserved[4];
} hi_vpss_grp_param;

typedef struct {
    hi_bool                 enable;            /* RW; range: [0, 1];  CROP enable. */
    hi_coord                crop_mode;   /* RW;  range: [0, 1]; coordinate mode of the crop start point. */
    hi_rect                 crop_rect;         /* CROP rectangular. */
} hi_vpss_crop_info;

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */

#endif /* __MPI_VPSS_H__ */
