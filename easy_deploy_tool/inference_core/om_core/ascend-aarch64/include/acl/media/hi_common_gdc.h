/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: Head of gdc
 * Author: Hisilicon multimedia software group
 * Create: 2023-01-07
 */
#ifndef HI_COMMON_GDC_H
#define HI_COMMON_GDC_H

#include "hi_media_type.h"
#include "hi_media_common.h"
#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

#define HI_FISHEYE_LMF_COEF_NUM            128
#define HI_FISHEYE_MAX_RGN_NUM             4

#define HI_SRC_LENS_COEF_SEG       2
#define HI_DST_LENS_COEF_SEG       3
#define HI_SRC_LENS_COEF_NUM       4
#define HI_DST_LENS_COEF_NUM       4
#define HI_DST_LENS_COEF_SEG_POINT (HI_DST_LENS_COEF_SEG - 1)

typedef enum hiVI_LDC_VERSION_E {
    HI_LDC_V1 = 1,
    HI_LDC_V2 = 2,
    HI_LDC_VERSION_BUTT
} hi_ldc_version;

typedef struct {
    /* RW;range: [0, 1];whether aspect ration  is keep */
    hi_bool    aspect;
    /* RW; range: [0, 100]; field angle ration of  horizontal,valid when aspect=0. */
    hi_s32     x_ratio;
    /* RW; range: [0, 100]; field angle ration of  vertical,valid when aspect=0. */
    hi_s32     y_ratio;
    /* RW; range: [0, 100]; field angle ration of  all,valid when aspect=1. */
    hi_s32     xy_ratio;
    /* RW; range: [-511, 511]; horizontal offset of the image distortion center relative to image center. */
    hi_s32 center_x_offset;
    /* RW; range: [-511, 511]; vertical offset of the image distortion center relative to image center. */
    hi_s32 center_y_offset;
    /* RW; range: [-300, 500]; LDC distortion ratio.when spread on,distortion_ratio range should be [0, 500] */
    hi_s32 distortion_ratio;
} hi_ldc_v1_attr;

typedef struct {
    /* RW; focal length in horizontal direction, with 2 decimal numbers */
    hi_s32 focal_len_x;
    /* RW; focal length in vertical direction, with 2 decimal numbers */
    hi_s32 focal_len_y;
    /* RW; coordinate of image center, with 2 decimal numbers */
    hi_s32 coor_shift_x;
    /* RW; Y coordinate of image center, with 2 decimal numbers */
    hi_s32 coor_shift_y;
    /* RW; lens distortion coefficients of the source image, with 5 decimal numbers */
    hi_s32 src_cali_ratio[HI_SRC_LENS_COEF_SEG][HI_SRC_LENS_COEF_NUM];
    /* RW; junction point of the two segments */
    hi_s32 src_jun_pt;
    /* RW; lens distortion coefficients, with 5 decimal numbers */
    hi_s32 dst_cali_ratio[HI_DST_LENS_COEF_SEG][HI_DST_LENS_COEF_NUM];
    /* RW; junction point of the three segments */
    hi_s32 dst_jun_pt[HI_DST_LENS_COEF_SEG_POINT];
    /* RW; max undistorted distance before 3rd polynomial drop, with 16bits decimal */
    hi_s32 max_du;
} hi_ldc_v2_attr;

typedef enum {
    HI_FISHEYE_MOUNT_MODE_DESKTOP = 0, /* Desktop mount mode */
    HI_FISHEYE_MOUNT_MODE_CEILING = 1, /* Ceiling mount mode */
    HI_FISHEYE_MOUNT_MODE_WALL = 2, /* wall mount mode */
    HI_FISHEYE_MOUNT_MODE_BUTT
} hi_fisheye_mount_mode;

typedef enum {
    HI_FISHEYE_VIEW_MODE_360_PANORAMA = 0, /* 360 panorama mode of gdc correction */
    HI_FISHEYE_VIEW_MODE_180_PANORAMA = 1, /* 180 panorama mode of gdc correction */
    HI_FISHEYE_VIEW_MODE_NORM = 2, /* normal mode of gdc correction */
    HI_FISHEYE_VIEW_MODE_NO_TRANS = 3, /* no gdc correction */
    HI_FISHEYE_VIEW_MODE_BUTT
} hi_fisheye_view_mode;

/* fisheye region correction attribute */
typedef struct {
    hi_fisheye_view_mode view_mode;
    hi_u32 in_radius;
    hi_u32 out_radius;
    hi_u32 pan;
    hi_u32 tilt;
    hi_u32 hor_zoom;
    hi_u32 ver_zoom;
    hi_rect out_rect;
} hi_fisheye_rgn_attr;

/* fisheye all regions correction attribute */
typedef struct {
    hi_bool lmf_en;
    hi_bool bg_color_en;
    hi_u32  bg_color;
    hi_s32  hor_offset;
    hi_s32  ver_offset;
    hi_u32  trapezoid_coef;
    hi_s32  fan_strength;
    hi_fisheye_mount_mode mount_mode;
    hi_u32 rgn_num;
    hi_fisheye_rgn_attr fisheye_rgn_attr[HI_FISHEYE_MAX_RGN_NUM];
} hi_fisheye_attr;

/* Fisheye Config */
typedef struct {
    hi_u16 lmf_coef[HI_FISHEYE_LMF_COEF_NUM]; /* RW;  LMF coefficient of gdc len */
} hi_fisheye_cfg;

typedef struct {
    hi_bool enable;
    hi_fisheye_attr fisheye_attr;
    hi_size dst_size;
} hi_fisheye_correction_attr;

/* spread correction attribute */
typedef struct {
    /* RW; range: [0, 1];whether enable spread or not, when spread on,ldc distortion_ratio range should be [0, 500] */
    hi_bool enable;
    hi_u32 spread_coef; /* RW; range: [0, 18];strength coefficient of spread correction */
} hi_spread_attr;

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */

#endif /* __MPI_COMM_GDC_H__ */