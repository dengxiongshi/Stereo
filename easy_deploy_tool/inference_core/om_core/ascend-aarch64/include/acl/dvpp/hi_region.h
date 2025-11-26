/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 and
 * only version 2 as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * Description:
 * Author: huawei
 * Create: 2023-4-11
 */

#ifndef HI_REGION_H_
#define HI_REGION_H_

#include "hi_dvpp_common.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif // #ifdef __cplusplus

#define HI_RGN_CLUT_NUM  16
#define HI_RGN_HANDLE_MAX 1024

/* invalid device ID */
#define HI_ERR_RGN_INVALID_DEV_ID 0xA0038001
/* invalid channel ID */
#define HI_ERR_RGN_INVALID_CHN_ID 0xA0038002
/* at least one parameter is illegal, e.g. an illegal enumeration value */
#define HI_ERR_RGN_ILLEGAL_PARAM  0xA0038003
/* channel exists */
#define HI_ERR_RGN_EXIST          0xA0038004
/* not exist */
#define HI_ERR_RGN_UNEXIST        0xA0038005
/* using a NULL pointer */
#define HI_ERR_RGN_NULL_PTR       0xA0038006
/* try to enable or initialize system, device or channel, before configuring attribute */
#define HI_ERR_RGN_NOT_CFG        0xA0038007
/* operation is not supported */
#define HI_ERR_RGN_NOT_SUPPORT    0xA0038008
/* operation is not permitted, e.g. try to change static attribute */
#define HI_ERR_RGN_NOT_PERM       0xA0038009
/* failure caused by malloc memory */
#define HI_ERR_RGN_NO_MEM         0xA003800C
/* failure caused by malloc buffer */
#define HI_ERR_RGN_NO_BUF         0xA003800D
/* no data in buffer */
#define HI_ERR_RGN_BUF_EMPTY      0xA003800E
/* no buffer for new data */
#define HI_ERR_RGN_BUF_FULL       0xA003800F
/* bad address, e.g. used for copy_from_user & copy_to_user */
#define HI_ERR_RGN_BAD_ADDR       0xA0038011
/* resource is busy, e.g. destroy a venc chn without unregistering it */
#define HI_ERR_RGN_BUSY           0xA0038012
/*
* System is not ready, maybe not initialized or loaded.
* Returning the error code when opening a device file failed.
*/
#define HI_ERR_RGN_NOT_READY      0xA0038010


typedef hi_u32 hi_rgn_handle;
typedef hi_u64 hi_phys_addr_t;

/* type of video regions */
typedef enum {
    HI_RGN_OVERLAY = 0,
    HI_RGN_COVER,
    HI_RGN_OVERLAYEX,
    HI_RGN_COVEREX,
    HI_RGN_LINE,
    HI_RGN_MOSAIC,
    HI_RGN_MOSAICEX,
    HI_RGN_CORNER_RECTEX,
} hi_rgn_type;

typedef struct {
    hi_pixel_format                pixel_format;
    hi_u32                         bg_color;    /* background color, pixel format depends on "pixel_format" */
    /*
     * region size, width:[2, HI_RGN_OVERLAY_MAX_WIDTH], align:2,
     * height:[2, HI_RGN_OVERLAY_MAX_HEIGHT], align:2
     */
    hi_size                        size;
    hi_u32                         canvas_num;
    hi_u32                         clut[HI_RGN_CLUT_NUM];
} hi_rgn_overlayex_attr;

typedef struct {
    /*
     * x:[0, HI_RGN_OVERLAYEX_MAX_X], align:2,
     * y:[0, HI_RGN_OVERLAYEX_MAX_Y], align:2
     */
    hi_point                       point;
    hi_u32                         fg_alpha;
    hi_u32                         bg_alpha;
    hi_u32                         layer;     /* OVERLAYEX region layer range depends on OVERLAYEX max num */
} hi_rgn_overlayex_chn_attr;

typedef struct {
    hi_cover                       cover;
    hi_u32                         layer;  /* COVER region layer range depends on COVER max num */
    hi_coord                       coord;  /* ratio coordinate or abs coordinate */
} hi_rgn_cover_chn_attr;

typedef struct {
    hi_rect                        rect;              /* location of MOSAIC */
    hi_blk_size                    blk_size;          /* block size of MOSAIC */
    hi_u32                         layer;             /* MOSAIC region layer range depends on MOSAIC max num */
} hi_rgn_mosaic_chn_attr;

typedef union {
    hi_rgn_overlayex_attr         overlayex; /* attribute of overlayex region. AUTO:hi_rgn_type:HI_RGN_OVERLAYEX; */
} hi_rgn_type_attr;


typedef union {
    hi_rgn_cover_chn_attr       cover_chn; /* attribute of cover region. AUTO:hi_rgn_type: HI_RGN_COVER; */
    hi_rgn_overlayex_chn_attr   overlayex_chn; /* attribute of overlayex region. AUTO:hi_rgn_type: HI_RGN_OVERLAYEX; */
    hi_rgn_mosaic_chn_attr      mosaic_chn; /* attribute of mosaic region. AUTO:hi_rgn_type:HI_RGN_MOSAIC; */
} hi_rgn_type_chn_attr;

/* attribute of a region */
typedef struct {
    hi_rgn_type                    type;
    hi_rgn_type_attr               attr; /* region attribute */
} hi_rgn_attr;

/* attribute of a region */
typedef struct {
    hi_bool                        is_show;
    hi_rgn_type                    type;
    hi_rgn_type_chn_attr           attr;        /* region attribute */
} hi_rgn_chn_attr;

typedef struct {
    hi_phys_addr_t                 phys_addr;
    hi_size                        size;
    hi_u32                         stride;
    hi_pixel_format                pixel_format;
    hi_void  ATTRIBUTE             *virt_addr;
} hi_rgn_canvas_info;


/*
 * @brief : create the region
 * @param [in] handle: handle of region [0, HI_RGN_HANDLE_MAX)
 * @param [in] rgn_attr: attribute of region
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_rgn_create(hi_rgn_handle handle, const hi_rgn_attr *rgn_attr);

/*
 * @brief : destroy the region
 * @param [in] handle: handle of region [0, HI_RGN_HANDLE_MAX)
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_rgn_destroy(hi_rgn_handle handle);

/*
 * @brief : apply the region to the channel
 * @param [in] handle: handle of region [0, HI_RGN_HANDLE_MAX)
 * @param [in] chn: struct of channel (module type, devid, channel id)
 * @param [in] chn_attr: attribute of the region acting on the channel
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_rgn_attach_to_chn(hi_rgn_handle handle, const hi_mpp_chn *chn, const hi_rgn_chn_attr *chn_attr);

/*
 * @brief : detach the region from the channel
 * @param [in] handle: handle of region [0, HI_RGN_HANDLE_MAX)
 * @param [in] chn: struct of channel (module type, devid, channel id)
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_rgn_detach_from_chn(hi_rgn_handle handle, const hi_mpp_chn *chn);

/*
 * @brief : set the display attribute of the region acting on the channel
 * @param [in] handle: handle of region [0, HI_RGN_HANDLE_MAX)
 * @param [in] chn: struct of channel (module type, devid, channel id)
 * @param [in] chn_attr: attribute of the region acting on the channel
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_rgn_set_display_attr(hi_rgn_handle handle, const hi_mpp_chn *chn, const hi_rgn_chn_attr *chn_attr);

/*
 * @brief : get the display attribute of the region acting on the channel
 * @param [in] handle: handle of region [0, HI_RGN_HANDLE_MAX)
 * @param [in] chn: struct of channel (module type, devid, channel id)
 * @param [out] chn_attr: attribute of the region acting on the channel
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_rgn_get_display_attr(hi_rgn_handle handle, const hi_mpp_chn *chn, hi_rgn_chn_attr *chn_attr);

/*
 * @brief : get the canvas info of the region
 * @param [in] handle: handle of region [0, HI_RGN_HANDLE_MAX)
 * @param [out] canvas_info: the canvas info of the region
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_rgn_get_canvas_info(hi_rgn_handle handle, hi_rgn_canvas_info *canvas_info);

/*
 * @brief : update the canvas info of the region after get canvas info and set canvas info
 * @param [in] handle: handle of region [0, HI_RGN_HANDLE_MAX)
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_rgn_update_canvas(hi_rgn_handle handle);


#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif // #ifdef __cplusplus

#endif // #ifndef HI_REGION_H_