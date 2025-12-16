/*
 * Copyright (C) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: Api of vi
 * Author: Hisilicon multimedia software group
 * Create: 2023/01/12
 */

#ifndef HI_MPI_VI_H
#define HI_MPI_VI_H

#include "hi_common_vi.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* __cplusplus */

/* 1 for vi device */
hi_s32 hi_mpi_vi_set_dev_attr(hi_vi_dev vi_dev, const hi_vi_dev_attr *dev_attr);
hi_s32 hi_mpi_vi_get_dev_attr(hi_vi_dev vi_dev, hi_vi_dev_attr *dev_attr);

hi_s32 hi_mpi_vi_enable_dev(hi_vi_dev vi_dev);
hi_s32 hi_mpi_vi_disable_dev(hi_vi_dev vi_dev);

hi_s32 hi_mpi_vi_set_dev_bind_pipe(hi_vi_dev vi_dev, const hi_vi_dev_bind_pipe *dev_bind_pipe);
hi_s32 hi_mpi_vi_get_dev_bind_pipe(hi_vi_dev vi_dev, hi_vi_dev_bind_pipe *dev_bind_pipe);

/* 2 for vi pipe */
hi_s32 hi_mpi_vi_destroy_pipe(hi_vi_pipe vi_pipe);
hi_s32 hi_mpi_vi_create_pipe(hi_vi_pipe vi_pipe, hi_vi_pipe_attr *pipe_attr);

hi_s32 hi_mpi_vi_start_pipe(hi_vi_pipe vi_pipe);
hi_s32 hi_mpi_vi_stop_pipe(hi_vi_pipe vi_pipe);

hi_s32 hi_mpi_vi_set_pipe_pre_crop(hi_vi_pipe vi_pipe, const hi_crop_info *crop_info);
hi_s32 hi_mpi_vi_get_pipe_pre_crop(hi_vi_pipe vi_pipe, hi_crop_info *crop_info);

hi_s32 hi_mpi_vi_set_pipe_frame_dump_attr(hi_vi_pipe vi_pipe, const hi_vi_frame_dump_attr *dump_attr);
hi_s32 hi_mpi_vi_get_pipe_frame_dump_attr(hi_vi_pipe vi_pipe, hi_vi_frame_dump_attr *dump_attr);

hi_s32 hi_mpi_vi_set_pipe_frame_source(hi_vi_pipe vi_pipe, const hi_vi_pipe_frame_source source);
hi_s32 hi_mpi_vi_get_pipe_frame_source(hi_vi_pipe vi_pipe, hi_vi_pipe_frame_source *source);

hi_s32 hi_mpi_vi_get_pipe_frame(hi_vi_pipe vi_pipe, hi_video_frame_info *frame_info, hi_s32 milli_sec);
hi_s32 hi_mpi_vi_release_pipe_frame(hi_vi_pipe vi_pipe, const hi_video_frame_info *frame_info);

hi_s32 hi_mpi_vi_send_pipe_raw(hi_u32 pipe_num, hi_vi_pipe pipe_id[],
    const hi_video_frame_info *frame_info[], hi_s32 milli_sec);
hi_s32 hi_mpi_vi_send_pipe_yuv(hi_vi_pipe vi_pipe, const hi_video_frame_info *frame_info, hi_s32 milli_sec);

hi_s32 hi_mpi_vi_pipe_release_buffer(hi_vi_pipe vi_pipe, const hi_video_frame_info *frame_info);
hi_s32 hi_mpi_vi_pipe_get_buffer(hi_vi_pipe vi_pipe, hi_video_frame_info *frame_info);

hi_s32 hi_mpi_vi_get_pipe_attr(hi_vi_pipe vi_pipe, hi_vi_pipe_attr *pipe_attr);

/* 3 for vi chn */
hi_s32 hi_mpi_vi_set_chn_attr(hi_vi_pipe vi_pipe, hi_vi_chn vi_chn, const hi_vi_chn_attr *chn_attr);
hi_s32 hi_mpi_vi_get_chn_attr(hi_vi_pipe vi_pipe, hi_vi_chn vi_chn, hi_vi_chn_attr *chn_attr);

hi_s32 hi_mpi_vi_disable_chn(hi_vi_pipe vi_pipe, hi_vi_chn vi_chn);
hi_s32 hi_mpi_vi_enable_chn(hi_vi_pipe vi_pipe, hi_vi_chn vi_chn);

hi_s32 hi_mpi_vi_get_chn_frame(hi_vi_pipe vi_pipe, hi_vi_chn vi_chn,
    hi_video_frame_info *frame_info, hi_s32 milli_sec);
hi_s32 hi_mpi_vi_release_chn_frame(hi_vi_pipe vi_pipe, hi_vi_chn vi_chn,
    const hi_video_frame_info *frame_info);

hi_s32 hi_mpi_vi_get_pipe_fd(hi_vi_pipe vi_pipe);
hi_s32 hi_mpi_vi_get_chn_fd(hi_vi_pipe vi_pipe, hi_vi_chn vi_chn);

hi_s32 hi_mpi_vi_set_chn_ldc_attr(hi_vi_pipe vi_pipe, hi_vi_chn vi_chn, const hi_vi_ldc_attr *ldc_attr);
hi_s32 hi_mpi_vi_get_chn_ldc_attr(hi_vi_pipe vi_pipe, hi_vi_chn vi_chn, hi_vi_ldc_attr *ldc_attr);

hi_s32 hi_mpi_vi_set_chn_dis_config(hi_vi_pipe vi_pipe, hi_vi_chn vi_chn, const hi_dis_config *dis_config);
hi_s32 hi_mpi_vi_get_chn_dis_config(hi_vi_pipe vi_pipe, hi_vi_chn vi_chn, hi_dis_config *dis_config);
hi_s32 hi_mpi_vi_set_chn_dis_attr(hi_vi_pipe vi_pipe, hi_vi_chn vi_chn, const hi_dis_attr *dis_attr);
hi_s32 hi_mpi_vi_get_chn_dis_attr(hi_vi_pipe vi_pipe, hi_vi_chn vi_chn, hi_dis_attr *dis_attr);
hi_s32 hi_mpi_vi_set_chn_dis_param(hi_vi_pipe vi_pipe, hi_vi_chn vi_chn, const hi_dis_param *dis_param);
hi_s32 hi_mpi_vi_get_chn_dis_param(hi_vi_pipe vi_pipe, hi_vi_chn vi_chn, hi_dis_param *dis_param);

hi_s32 hi_mpi_vi_set_chn_low_delay_attr(hi_vi_pipe vi_pipe, hi_vi_chn vi_chn,
    const hi_vi_low_delay_info *delay_info);
hi_s32 hi_mpi_vi_get_chn_low_delay_attr(hi_vi_pipe vi_pipe, hi_vi_chn vi_chn,
    hi_vi_low_delay_info *delay_info);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */

#endif /* __MPI_VI_H__ */
