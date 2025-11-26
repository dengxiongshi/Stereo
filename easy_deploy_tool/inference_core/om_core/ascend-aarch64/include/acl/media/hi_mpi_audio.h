/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2021-2023. All rights reserved.
 * Description: multimedia common file
 * Author: Hisilicon multimedia software group
 * Create: 2021/04/27
 */

#ifndef HI_MPI_AUDIO_H
#define HI_MPI_AUDIO_H

#include "hi_common_aio.h"
#include "hi_common_aenc.h"
#include "hi_common_adec.h"

#ifdef __cplusplus
extern "C" {
#endif

/* AI function api. */
hi_s32 hi_mpi_ai_set_pub_attr(hi_audio_dev ai_dev, hi_aio_attr *attr);

hi_s32 hi_mpi_ai_enable(hi_audio_dev ai_dev);
hi_s32 hi_mpi_ai_disable(hi_audio_dev ai_dev);

hi_s32 hi_mpi_ai_enable_chn(hi_audio_dev ai_dev, hi_ai_chn ai_chn);
hi_s32 hi_mpi_ai_disable_chn(hi_audio_dev ai_dev, hi_ai_chn ai_chn);

hi_s32 hi_mpi_ai_enable_resample(hi_audio_dev ai_dev, hi_ai_chn ai_chn, hi_audio_sample_rate out_sample_rate);
hi_s32 hi_mpi_ai_disable_resample(hi_audio_dev ai_dev, hi_ai_chn ai_chn);

hi_s32 hi_mpi_ai_get_frame(hi_audio_dev ai_dev, hi_ai_chn ai_chn,
                           hi_audio_frame *frame, hi_aec_frame *aec_frame, hi_s32 milli_sec);
hi_s32 hi_mpi_ai_release_frame(hi_audio_dev ai_dev, hi_ai_chn ai_chn,
                               const hi_audio_frame *frame, const hi_aec_frame *aec_frame);

hi_s32 hi_mpi_ai_set_chn_attr(hi_audio_dev ai_dev, hi_ai_chn ai_chn, const hi_ai_chn_attr *chn_attr);

/* AO function api. */
hi_s32 hi_mpi_ao_set_pub_attr(hi_audio_dev ao_dev, hi_aio_attr *attr);

hi_s32 hi_mpi_ao_enable(hi_audio_dev ao_dev);
hi_s32 hi_mpi_ao_disable(hi_audio_dev ao_dev);

hi_s32 hi_mpi_ao_enable_chn(hi_audio_dev ao_dev, hi_ao_chn ao_chn);
hi_s32 hi_mpi_ao_disable_chn(hi_audio_dev ao_dev, hi_ao_chn ao_chn);

hi_s32 hi_mpi_ao_enable_resample(hi_audio_dev ao_dev, hi_ao_chn ao_chn, hi_audio_sample_rate in_sample_rate);
hi_s32 hi_mpi_ao_disable_resample(hi_audio_dev ao_dev, hi_ao_chn ao_chn);

hi_s32 hi_mpi_ao_send_frame(hi_audio_dev ao_dev, hi_ao_chn ao_chn, const hi_audio_frame *data, hi_s32 milli_sec);

hi_s32 hi_mpi_ao_get_chn_delay(hi_audio_dev ao_dev, hi_ao_chn ao_chn, hi_u32 *milli_sec);

/* AENC function api. */
hi_s32 hi_mpi_aenc_create_chn(hi_aenc_chn aenc_chn, const hi_aenc_chn_attr *attr);
hi_s32 hi_mpi_aenc_destroy_chn(hi_aenc_chn aenc_chn);

hi_s32 hi_mpi_aenc_get_stream(hi_aenc_chn aenc_chn, hi_audio_stream *stream, hi_s32 milli_sec);
hi_s32 hi_mpi_aenc_release_stream(hi_aenc_chn aenc_chn, const hi_audio_stream *stream);

/* ADEC function api. */
hi_s32 hi_mpi_adec_create_chn(hi_adec_chn adec_chn, const hi_adec_chn_attr *attr);
hi_s32 hi_mpi_adec_destroy_chn(hi_adec_chn adec_chn);

hi_s32 hi_mpi_adec_send_stream(hi_adec_chn adec_chn, const hi_audio_stream *stream, hi_bool block);

#ifdef __cplusplus
}
#endif
#endif /* HI_MPI_AUDIO_H */
