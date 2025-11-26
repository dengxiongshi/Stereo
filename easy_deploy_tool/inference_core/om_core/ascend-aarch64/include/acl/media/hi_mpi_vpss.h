/*
 * Copyright (C) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: Api of vpss
 * Author: Hisilicon multimedia software group
 * Create: 2023/01/12
 */

#ifndef HI_MPI_VPSS_H
#define HI_MPI_VPSS_H

#include "hi_common_vpss.h"
#include "hi_common_sns.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* __cplusplus */

/* group settings */
hi_s32 hi_mpi_vpss_create_grp(hi_vpss_grp grp, const hi_vpss_grp_attr *grp_attr);
hi_s32 hi_mpi_vpss_destroy_grp(hi_vpss_grp grp);

hi_s32 hi_mpi_vpss_start_grp(hi_vpss_grp grp);
hi_s32 hi_mpi_vpss_stop_grp(hi_vpss_grp grp);

hi_s32 hi_mpi_vpss_set_grp_crop(hi_vpss_grp grp, const hi_vpss_crop_info *crop_info);
hi_s32 hi_mpi_vpss_get_grp_crop(hi_vpss_grp grp, hi_vpss_crop_info *crop_info);

hi_s32 hi_mpi_vpss_set_grp_fisheye_cfg(hi_vpss_grp grp, const hi_fisheye_cfg *fisheye_cfg);
hi_s32 hi_mpi_vpss_get_grp_fisheye_cfg(hi_vpss_grp grp, hi_fisheye_cfg *fisheye_cfg);

/* chn settings */
hi_s32 hi_mpi_vpss_set_chn_attr(hi_vpss_grp grp, hi_vpss_chn chn, const hi_vpss_chn_attr *chn_attr);
hi_s32 hi_mpi_vpss_get_chn_attr(hi_vpss_grp grp, hi_vpss_chn chn, hi_vpss_chn_attr *chn_attr);

hi_s32 hi_mpi_vpss_enable_chn(hi_vpss_grp grp, hi_vpss_chn chn);
hi_s32 hi_mpi_vpss_disable_chn(hi_vpss_grp grp, hi_vpss_chn chn);

hi_s32 hi_mpi_vpss_get_chn_frame(hi_vpss_grp grp, hi_vpss_chn chn,
    hi_video_frame_info *frame_info, hi_s32 milli_sec);
hi_s32 hi_mpi_vpss_release_chn_frame(hi_vpss_grp grp, hi_vpss_chn chn,
    const hi_video_frame_info *frame_info);

/* 3DNR */
hi_s32 hi_mpi_vpss_set_grp_param(hi_vpss_grp grp, const hi_vpss_grp_param *grp_param);
hi_s32 hi_mpi_vpss_get_grp_param(hi_vpss_grp grp, hi_vpss_grp_param *grp_param);
/* fisheye */
hi_s32 hi_mpi_vpss_set_chn_fisheye(hi_vpss_grp grp, hi_vpss_chn chn, const hi_fisheye_correction_attr *correction_attr);
hi_s32 hi_mpi_vpss_get_chn_fisheye(hi_vpss_grp grp, hi_vpss_chn chn, hi_fisheye_correction_attr *correction_attr);

hi_s32 hi_mpi_vpss_get_chn_fd(hi_vpss_grp grp, hi_vpss_chn chn);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */

#endif /* __MPI_VI_H__ */
