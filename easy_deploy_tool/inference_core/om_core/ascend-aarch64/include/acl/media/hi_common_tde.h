/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: tde API common header file
 * Author: Hisilicon multimedia software group
 * Create: 2023/03/10
 */

#ifndef HI_COMMON_TDE_H
#define HI_COMMON_TDE_H

#ifdef __cplusplus
extern "C" {
#endif

#define HI_ERR_TDE_BASE (int)(((0x80UL + 0x20UL) << 24U) | (100U << 16U) | (4U << 13U) | 1U)

typedef enum {
    HI_ERR_TDE_DEV_NOT_OPEN = HI_ERR_TDE_BASE, /* tde device not open yet */
    HI_ERR_TDE_DEV_OPEN_FAILED,                /* open tde device failed */
    HI_ERR_TDE_NULL_PTR,                       /* input parameters contain null ptr */
    HI_ERR_TDE_NO_MEM,                         /* malloc failed  */
    HI_ERR_TDE_INVALID_HANDLE,                 /* invalid job handle */
    HI_ERR_TDE_INVALID_PARAM,                  /* invalid parameter */
    HI_ERR_TDE_NOT_ALIGNED,                    /* aligned error for position, stride, width */
    HI_ERR_TDE_MINIFICATION,                   /* invalid minification */
    HI_ERR_TDE_CLIP_AREA,                      /* clip area and operation area have no intersection */
    HI_ERR_TDE_JOB_TIMEOUT,                    /* blocked job wait timeout */
    HI_ERR_TDE_UNSUPPORTED_OPERATION,          /* unsupported operation */
    HI_ERR_TDE_QUERY_TIMEOUT,                  /* query time out */
    HI_ERR_TDE_INTERRUPT,                      /* blocked job was interrupted */
    HI_ERR_TDE_BUTT,
} hi_tde_err_code;

/* RGB and packet YUV formats and semi-planar YUV format */
typedef enum {
    HI_TDE_COLOR_FORMAT_RGB444 = 0, /* RGB444 format */
    HI_TDE_COLOR_FORMAT_BGR444,     /* BGR444 format */
    HI_TDE_COLOR_FORMAT_RGB555,     /* RGB555 format */
    HI_TDE_COLOR_FORMAT_BGR555,     /* BGR555 format */
    HI_TDE_COLOR_FORMAT_RGB565,     /* RGB565 format */
    HI_TDE_COLOR_FORMAT_BGR565,     /* BGR565 format */
    HI_TDE_COLOR_FORMAT_RGB888,     /* RGB888 format */
    HI_TDE_COLOR_FORMAT_BGR888,     /* BGR888 format */
    HI_TDE_COLOR_FORMAT_ARGB4444,   /* ARGB4444 format */
    HI_TDE_COLOR_FORMAT_ABGR4444,   /* ABGR4444 format */
    HI_TDE_COLOR_FORMAT_RGBA4444,   /* RGBA4444 format */
    HI_TDE_COLOR_FORMAT_BGRA4444,   /* BGRA4444 format */
    HI_TDE_COLOR_FORMAT_ARGB1555,   /* ARGB1555 format */
    HI_TDE_COLOR_FORMAT_ABGR1555,   /* ABGR1555 format */
    HI_TDE_COLOR_FORMAT_RGBA1555,   /* RGBA1555 format */
    HI_TDE_COLOR_FORMAT_BGRA1555,   /* BGRA1555 format */
    HI_TDE_COLOR_FORMAT_ARGB8565,   /* ARGB8565 format */
    HI_TDE_COLOR_FORMAT_ABGR8565,   /* ABGR8565 format */
    HI_TDE_COLOR_FORMAT_RGBA8565,   /* RGBA8565 format */
    HI_TDE_COLOR_FORMAT_BGRA8565,   /* BGRA8565 format */
    HI_TDE_COLOR_FORMAT_ARGB8888,   /* ARGB8888 format */
    HI_TDE_COLOR_FORMAT_ABGR8888,   /* ABGR8888 format */
    HI_TDE_COLOR_FORMAT_RGBA8888,   /* RGBA8888 format */
    HI_TDE_COLOR_FORMAT_BGRA8888,   /* BGRA8888 format */
    HI_TDE_COLOR_FORMAT_RABG8888,   /* RABG8888 format */
    /* 1-bit palette format without alpha component. Each pixel occupies one bit. */
    HI_TDE_COLOR_FORMAT_CLUT1,
    /* 2-bit palette format without alpha component. Each pixel occupies two bits. */
    HI_TDE_COLOR_FORMAT_CLUT2,
    /* 4-bit palette format without alpha component. Each pixel occupies four bits. */
    HI_TDE_COLOR_FORMAT_CLUT4,
    /* 8-bit palette format without alpha component. Each pixel occupies eight bits. */
    HI_TDE_COLOR_FORMAT_CLUT8,
    /* 4-bit palette format with alpha component. Each pixel occupies 8 bit. */
    HI_TDE_COLOR_FORMAT_ACLUT44,
    /* 8-bit palette format with alpha component. Each pixel occupies 16 bit. */
    HI_TDE_COLOR_FORMAT_ACLUT88,
    HI_TDE_COLOR_FORMAT_A1,               /* Alpha format. Each pixel occupies one bit. */
    HI_TDE_COLOR_FORMAT_A8,               /* Alpha format. Each pixel occupies eight bits. */
    HI_TDE_COLOR_FORMAT_YCbCr888,         /* YUV packet format without alpha component */
    HI_TDE_COLOR_FORMAT_AYCbCr8888,       /* YUV packet format with alpha component */
    HI_TDE_COLOR_FORMAT_YCbCr422,         /* YUV packet422 format */
    HI_TDE_COLOR_FORMAT_PKGVYUY,          /* YUV packet422 format, VYUY format */
    HI_TDE_COLOR_FORMAT_BYTE,             /* Only for fast copy */
    HI_TDE_COLOR_FORMAT_HALFWORD,         /* Only for fast copy */
    HI_TDE_COLOR_FORMAT_JPG_YCbCr400MBP,  /* Semi-planar YUV400 format, for JPG decoding */
    HI_TDE_COLOR_FORMAT_JPG_YCbCr422MBHP, /* Semi-planar YUV422 format, horizontal sampling, for JPG decoding */
    HI_TDE_COLOR_FORMAT_JPG_YCbCr422MBVP, /* Semi-planar YUV422 format, vertical sampling, for JPG decoding */
    HI_TDE_COLOR_FORMAT_MP1_YCbCr420MBP,  /* Semi-planar YUV420 format */
    HI_TDE_COLOR_FORMAT_MP2_YCbCr420MBP,  /* Semi-planar YUV420 format */
    HI_TDE_COLOR_FORMAT_MP2_YCbCr420MBI,  /* Semi-planar YUV420 format */
    HI_TDE_COLOR_FORMAT_JPG_YCbCr420MBP,  /* Semi-planar YUV420 format, for JPG decoding */
    HI_TDE_COLOR_FORMAT_JPG_YCbCr444MBP,  /* Semi-planar YUV444 format */
    HI_TDE_COLOR_FORMAT_MAX               /* End of enumeration */
} hi_tde_color_format;

/* Definition of alpha output sources */
typedef enum {
    HI_TDE_OUT_ALPHA_FROM_NORM = 0,    /* Output from the result of alpha blending or anti-flicker */
    HI_TDE_OUT_ALPHA_FROM_BG,          /* Output from the background bitmap */
    HI_TDE_OUT_ALPHA_FROM_FG,          /* Output from the foreground bitmap */
    HI_TDE_OUT_ALPHA_FROM_GLOBALALPHA, /* Output from the global alpha */
    HI_TDE_OUT_ALPHA_FROM_MAX
} hi_tde_out_alpha_from;

/* blend mode */
typedef enum {
    HI_TDE_BLEND_ZERO = 0x0,
    HI_TDE_BLEND_ONE,
    HI_TDE_BLEND_SRC2COLOR,
    HI_TDE_BLEND_INVSRC2COLOR,
    HI_TDE_BLEND_SRC2ALPHA,
    HI_TDE_BLEND_INVSRC2ALPHA,
    HI_TDE_BLEND_SRC1COLOR,
    HI_TDE_BLEND_INVSRC1COLOR,
    HI_TDE_BLEND_SRC1ALPHA,
    HI_TDE_BLEND_INVSRC1ALPHA,
    HI_TDE_BLEND_SRC2ALPHASAT,
    HI_TDE_BLEND_MAX
} hi_tde_blend_mode;

/* Alpha blending command. You can set parameters or select Porter or Duff. */
/* pixel = (source * fs + destination * fd),
   sa = source alpha,
   da = destination alpha */
typedef enum {
    HI_TDE_BLEND_CMD_NONE = 0x0, /* fs: sa      fd: 1.0-sa */
    HI_TDE_BLEND_CMD_CLEAR,      /* fs: 0.0     fd: 0.0 */
    HI_TDE_BLEND_CMD_SRC,        /* fs: 1.0     fd: 0.0 */
    HI_TDE_BLEND_CMD_SRCOVER,    /* fs: 1.0     fd: 1.0-sa */
    HI_TDE_BLEND_CMD_DSTOVER,    /* fs: 1.0-da  fd: 1.0 */
    HI_TDE_BLEND_CMD_SRCIN,      /* fs: da      fd: 0.0 */
    HI_TDE_BLEND_CMD_DSTIN,      /* fs: 0.0     fd: sa */
    HI_TDE_BLEND_CMD_SRCOUT,     /* fs: 1.0-da  fd: 0.0 */
    HI_TDE_BLEND_CMD_DSTOUT,     /* fs: 0.0     fd: 1.0-sa */
    HI_TDE_BLEND_CMD_SRCATOP,    /* fs: da      fd: 1.0-sa */
    HI_TDE_BLEND_CMD_DSTATOP,    /* fs: 1.0-da  fd: sa */
    HI_TDE_BLEND_CMD_ADD,        /* fs: 1.0     fd: 1.0 */
    HI_TDE_BLEND_CMD_XOR,        /* fs: 1.0-da  fd: 1.0-sa */
    HI_TDE_BLEND_CMD_DST,        /* fs: 0.0     fd: 1.0 */
    HI_TDE_BLEND_CMD_CONFIG,     /* You can set the parameteres. */
    HI_TDE_BLEND_CMD_MAX
} hi_tde_blend_cmd;

/* Logical operation type */
typedef enum {
    HI_TDE_ALPHA_BLENDING_NONE = 0x0,     /* No alpha and raster of operation (ROP) blending */
    HI_TDE_ALPHA_BLENDING_BLEND = 0x1,    /* Alpha blending */
    HI_TDE_ALPHA_BLENDING_ROP = 0x2,      /* ROP blending */
    HI_TDE_ALPHA_BLENDING_COLORIZE = 0x4, /* Colorize operation */
    HI_TDE_ALPHA_BLENDING_MAX = 0x8       /* End of enumeration */
} hi_tde_alpha_blending;

/* Definition of ROP codes */
typedef enum {
    HI_TDE_ROP_BLACK = 0,   /* Blackness */
    HI_TDE_ROP_NOTMERGEPEN, /* ~(S2 | S1) */
    HI_TDE_ROP_MASKNOTPEN,  /* ~S2&S1 */
    HI_TDE_ROP_NOTCOPYPEN,  /* ~S2 */
    HI_TDE_ROP_MASKPENNOT,  /* S2&~S1 */
    HI_TDE_ROP_NOT,         /* ~S1 */
    HI_TDE_ROP_XORPEN,      /* S2^S1 */
    HI_TDE_ROP_NOTMASKPEN,  /* ~(S2 & S1) */
    HI_TDE_ROP_MASKPEN,     /* S2&S1 */
    HI_TDE_ROP_NOTXORPEN,   /* ~(S2^S1) */
    HI_TDE_ROP_NOP,         /* S1 */
    HI_TDE_ROP_MERGENOTPEN, /* ~S2|S1 */
    HI_TDE_ROP_COPYPEN,     /* S2 */
    HI_TDE_ROP_MERGEPENNOT, /* S2|~S1 */
    HI_TDE_ROP_MERGEPEN,    /* S2|S1 */
    HI_TDE_ROP_WHITE,       /* Whiteness */
    HI_TDE_ROP_MAX
} hi_tde_rop_mode;

/* Clip operation type */
typedef enum {
    HI_TDE_CLIP_MODE_NONE = 0, /* No clip */
    HI_TDE_CLIP_MODE_INSIDE,   /* Clip the data within the rectangle to output and discard others */
    HI_TDE_CLIP_MODE_OUTSIDE,  /* Clip the data outside the rectangle to output and discard others */
    HI_TDE_CLIP_MODE_MAX
} hi_tde_clip_mode;

/* Definition of colorkey modes */
typedef enum {
    HI_TDE_COLORKEY_MODE_NONE = 0, /* No colorkey */
    /* When performing the colorkey operation on the foreground bitmap,
    you need to perform this operation before the CLUT for color extension
    and perform this operation after the CLUT for color correction. */
    HI_TDE_COLORKEY_MODE_FG,
    HI_TDE_COLORKEY_MODE_BG, /* Perform the colorkey operation on the background bitmap */
    HI_TDE_COLORKEY_MODE_MAX
} hi_tde_colorkey_mode;

#ifdef __cplusplus
}
#endif
#endif /* HI_COMMON_TDE_H */
