/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: multimedia common file
 * Author: Hisilicon multimedia software group
 * Create: 2023/03/19
 */

#ifndef HI_FB_H
#define HI_FB_H

#include "hi_media_type.h"

#ifdef __cplusplus
extern "C" {
#endif

#define IOC_TYPE_HIFB               'F'
/* To get the alpha of an overlay layer */
#define FBIOGET_ALPHA_HIFB          _IOR(IOC_TYPE_HIFB, 92, hi_fb_alpha)
/* To set the alpha of an overlay layer */
#define FBIOPUT_ALPHA_HIFB          _IOW(IOC_TYPE_HIFB, 93, hi_fb_alpha)
/* To get the origin of an overlay layer on the screen */
#define FBIOGET_SCREEN_ORIGIN_HIFB  _IOR(IOC_TYPE_HIFB, 94, hi_fb_point)
/* To set the origin of an overlay layer on the screen */
#define FBIOPUT_SCREEN_ORIGIN_HIFB  _IOW(IOC_TYPE_HIFB, 95, hi_fb_point)
/* To set the display state of an overlay layer */
#define FBIOPUT_SHOW_HIFB           _IOW(IOC_TYPE_HIFB, 101, hi_bool)
/* To obtain the display state of an overlay layer */
#define FBIOGET_SHOW_HIFB           _IOR(IOC_TYPE_HIFB, 102, hi_bool)

/* Origin Point info */
typedef struct {
    hi_s32 x_pos;         /* <  horizontal position */
    hi_s32 y_pos;         /* <  vertical position */
} hi_fb_point;

/* Alpha info */
typedef struct {
    hi_bool alpha_en;   /*  pixel alpha enable flag */
    hi_bool alpha_chn_en;  /*  global alpha enable flag */
    hi_u8 alpha0;         /*  alpha0 value, used in ARGB1555 */
    hi_u8 alpha1;         /*  alpha1 value, used in ARGB1555 */
    hi_u8 global_alpha;    /*  global alpha value */
    hi_u8 reserved;
} hi_fb_alpha;

#ifdef __cplusplus
}
#endif
#endif /* HI_FB_H */