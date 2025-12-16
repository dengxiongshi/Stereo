/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: tde API base header file
 * Author: Hisilicon multimedia software group
 * Create: 2023/03/10
 */

#ifndef HI_BASE_TDE_H
#define HI_BASE_TDE_H

#include "hi_media_type.h"
#include "hi_media_common.h"
#include "hi_common_tde.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Structure of the bitmap information set by customers */
typedef struct {
    hi_u64 phys_addr; /* Header address of a bitmap or the Y component */
    hi_u32 phys_len;
    hi_tde_color_format color_format; /* Color format */
    hi_u32 height;                    /* Bitmap height */
    hi_u32 width;                     /* Bitmap width */
    hi_u32 stride;                    /* Stride of a bitmap or the Y component */
    hi_bool is_ycbcr_clut;            /* Whether the CLUT is in the YCbCr space. */
    hi_bool alpha_max_is_255;         /* The maximum alpha value of a bitmap is 255 or 128. */
    hi_bool support_alpha_ex_1555;    /* Whether to enable the alpha extension of an ARGB1555 bitmap. */
    hi_u8 alpha0;                     /* Values of alpha0 and alpha1, used as the ARGB1555 format */
    hi_u8 alpha1;                     /* Values of alpha0 and alpha1, used as the ARGB1555 format */
    hi_u64 cbcr_phys_addr;            /* Address of the CbCr component, pilot */
    hi_u32 cbcr_phys_len;
    hi_u32 cbcr_stride; /* Stride of the CbCr component, pilot */
    /* <Address of the color look-up table (CLUT), for color extension or color correction */
    hi_u64 clut_phys_addr;
    hi_u32 clut_phys_len;
} hi_tde_surface;

/* Definition of the TDE rectangle */
typedef struct {
    hi_s32 pos_x;  /* Horizontal coordinate */
    hi_s32 pos_y;  /* Vertical coordinate */
    hi_u32 width;  /* Width */
    hi_u32 height; /* Height */
} hi_tde_rect;

/* dma module */
typedef struct {
    hi_tde_surface *dst_surface;
    hi_tde_rect *dst_rect;
} hi_tde_none_src;

/* single source */
typedef struct {
    hi_tde_surface *src_surface;
    hi_tde_surface *dst_surface;
    hi_tde_rect *src_rect;
    hi_tde_rect *dst_rect;
} hi_tde_single_src;

/* double source */
typedef struct {
    hi_tde_surface *bg_surface;
    hi_tde_surface *fg_surface;
    hi_tde_surface *dst_surface;
    hi_tde_rect *bg_rect;
    hi_tde_rect *fg_rect;
    hi_tde_rect *dst_rect;
} hi_tde_double_src;

/* Definition of fill colors */
typedef struct {
    hi_tde_color_format color_format; /* TDE pixel format */
    hi_u32 color_value;               /* Fill colors that vary according to pixel formats */
} hi_tde_fill_color;

/* Definition of colorkey range */
typedef struct {
    hi_u8 min_component;       /* Minimum value of a component */
    hi_u8 max_component;       /* Maximum value of a component */
    hi_u8 is_component_out;    /* The colorkey of a component is within or beyond the range. */
    hi_u8 is_component_ignore; /* Whether to ignore a component. */
    hi_u8 component_mask;      /* Component mask */
    hi_u8 component_reserved;
    hi_u8 component_reserved1;
    hi_u8 component_reserved2;
} hi_tde_colorkey_component;

/* Definition of colorkey values */
typedef union {
    struct {
        hi_tde_colorkey_component alpha; /* Alpha component */
        hi_tde_colorkey_component red;   /* Red component */
        hi_tde_colorkey_component green; /* Green component */
        hi_tde_colorkey_component blue;  /* Blue component */
    } argb_colorkey;                     /* AUTO:hi_tde_colorkey_mode:HI_TDE_COLORKEY_MODE_NONE; */
    struct {
        hi_tde_colorkey_component alpha; /* Alpha component */
        hi_tde_colorkey_component y;     /* Y component */
        hi_tde_colorkey_component cb;    /* Cb component */
        hi_tde_colorkey_component cr;    /* Cr component */
    } ycbcr_colorkey;                    /* AUTO:hi_tde_colorkey_mode:HI_TDE_COLORKEY_MODE_FG; */
    struct {
        hi_tde_colorkey_component alpha; /* Alpha component */
        hi_tde_colorkey_component clut;  /* Palette component */
    } clut_colorkey;                     /* AUTO:hi_tde_colorkey_mode:HI_TDE_COLORKEY_MODE_BG; */
} hi_tde_colorkey;

/* Options for the alpha blending operation */
typedef struct {
    hi_bool global_alpha_en;     /* Global alpha enable */
    hi_bool pixel_alpha_en;      /* Pixel alpha enable */
    hi_bool src1_alpha_premulti; /* Src1 alpha premultiply enable */
    hi_bool src2_alpha_premulti; /* Src2 alpha premultiply enable */
    hi_tde_blend_cmd blend_cmd;  /* Alpha blending command */
    /* Src1 blending mode select. It is valid when blend_cmd is set to HI_TDE_BLEND_CMD_CONFIG. */
    hi_tde_blend_mode src1_blend_mode;
    /* Src2 blending mode select. It is valid when blend_cmd is set to HI_TDE_BLEND_CMD_CONFIG. */
    hi_tde_blend_mode src2_blend_mode;
} hi_tde_blend_opt;

/* CSC parameter option */
typedef struct {
    hi_bool src_csc_user_en;         /* User-defined ICSC parameter enable */
    hi_bool src_csc_param_reload_en; /* User-defined ICSC parameter reload enable */
    hi_bool dst_csc_user_en;         /* User-defined OCSC parameter enable */
    hi_bool dst_csc_param_reload_en; /* User-defined OCSC parameter reload enable */
    hi_u64 src_csc_param_addr;       /* ICSC parameter address. The address must be 128-bit aligned. */
    hi_s32 src_csc_param_len;
    hi_u64 dst_csc_param_addr; /* OCSC parameter address. The address must be 128-bit aligned. */
    hi_s32 dst_csc_param_len;
} hi_tde_csc_opt;

/* Definition of the pattern filling operation */
typedef struct {
    hi_tde_alpha_blending alpha_blending_cmd; /* Logical operation type */
    hi_tde_rop_mode rop_color;                /* ROP type of the color space */
    hi_tde_rop_mode rop_alpha;                /* ROP type of the alpha component */
    hi_tde_colorkey_mode colorkey_mode;       /* Colorkey mode */
    hi_tde_colorkey colorkey_value;           /* Colorkey value */
    hi_tde_clip_mode clip_mode;               /* Clip mode */
    hi_tde_rect clip_rect;                    /* Clipping area */
    hi_bool clut_reload;                      /* Whether to reload the CLUT */
    hi_u8 global_alpha;                       /* Global alpha */
    hi_tde_out_alpha_from out_alpha_from;     /* Source of the output alpha */
    hi_u32 color_resize;                      /* Colorize value */
    hi_tde_blend_opt blend_opt;               /* Options of the blending operation */
    hi_tde_csc_opt csc_opt;                   /* CSC parameter option */
} hi_tde_pattern_fill_opt;

#ifdef __cplusplus
}
#endif
#endif /* HI_BASE_TDE_H */
