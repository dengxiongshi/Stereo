/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: vo common file
 * Author: Hisilicon multimedia software group
 * Create: 2023/03/19
 */

#ifndef __HI_MPI_VO_H__
#define __HI_MPI_VO_H__

#include "hi_common_vo.h"

#ifdef __cplusplus
extern "C" {
#endif

hi_s32 hi_mpi_vo_set_pub_attr(hi_vo_dev dev, const hi_vo_pub_attr *pub_attr);
hi_s32 hi_mpi_vo_enable(hi_vo_dev dev);
hi_s32 hi_mpi_vo_disable(hi_vo_dev dev);
hi_s32 hi_mpi_vo_set_video_layer_attr(hi_vo_layer layer, const hi_vo_video_layer_attr *layer_attr);
hi_s32 hi_mpi_vo_get_video_layer_attr(hi_vo_layer layer, hi_vo_video_layer_attr *layer_attr);
hi_s32 hi_mpi_vo_enable_video_layer(hi_vo_layer layer);
hi_s32 hi_mpi_vo_disable_video_layer(hi_vo_layer layer);
hi_s32 hi_mpi_vo_enable_chn(hi_vo_layer layer, hi_vo_chn chn);
hi_s32 hi_mpi_vo_disable_chn(hi_vo_layer layer, hi_vo_chn chn);
hi_s32 hi_mpi_vo_set_chn_attr(hi_vo_layer layer, hi_vo_chn chn, const hi_vo_chn_attr *chn_attr);
hi_s32 hi_mpi_vo_set_chn_param(hi_vo_layer layer, hi_vo_chn chn, const hi_vo_chn_param *chn_param);
hi_s32 hi_mpi_vo_send_frame(hi_vo_layer layer, hi_vo_chn chn, const hi_video_frame_info *frame_info,
    hi_s32 milli_sec);
hi_s32 hi_mpi_vo_pause_chn(int layer, int chn);
hi_s32 hi_mpi_vo_resume_chn(int layer, int chn);
hi_s32 hi_mpi_vo_create_pool(const hi_u64 size);
hi_u64 hi_mpi_vo_handle_to_phys_addr(hi_s32 handle);
hi_s32 hi_mpi_vo_destroy_pool(hi_s32 handle);
hi_s32 hi_mpi_vo_hide_chn(hi_vo_layer layer, hi_vo_chn chn);
hi_s32 hi_mpi_vo_show_chn(hi_vo_layer layer, hi_vo_chn chn);
hi_s32 hi_mpi_vo_set_chn_frame_rate(hi_vo_layer layer, hi_vo_layer chn, hi_s32 frame_rate);
hi_s32 hi_mpi_vo_bind_layer(hi_vo_layer layer, hi_vo_dev dev);
hi_s32 hi_mpi_vo_unbind_layer(hi_vo_layer layer, hi_vo_dev dev);
#ifdef __cplusplus
}
#endif
#endif /* __HI_MPI_VO_H__ */
