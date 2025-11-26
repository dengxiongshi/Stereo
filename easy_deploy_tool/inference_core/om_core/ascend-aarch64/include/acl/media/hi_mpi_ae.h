/*
 * Copyright (C) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: Api of ae
 * Author: Hisilicon multimedia software group
 * Create: 2023/01/12
 */

#ifndef HI_MPI_AE_H
#define HI_MPI_AE_H

#include "hi_common_ae.h"
#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* __cplusplus */
/* The interface of ae lib register to isp. */
hi_s32 hi_mpi_ae_register(hi_vi_pipe vi_pipe, hi_isp_3a_alg_lib *ae_lib);
hi_s32 hi_mpi_ae_unregister(hi_vi_pipe vi_pipe, hi_isp_3a_alg_lib *ae_lib);

/* The callback function of sensor register to ae lib. */
hi_s32 hi_mpi_ae_sensor_reg_callback(hi_vi_pipe vi_pipe, hi_isp_3a_alg_lib *ae_lib,
    const hi_isp_sns_attr_info *sns_attr_info, const hi_isp_ae_sensor_register *pregister);
hi_s32 hi_mpi_ae_sensor_unreg_callback(hi_vi_pipe vi_pipe, hi_isp_3a_alg_lib *ae_lib, hi_sensor_id sensor_id);

hi_s32 hi_mpi_isp_set_exposure_attr(hi_vi_pipe vi_pipe, const hi_isp_exposure_attr *exp_attr);
hi_s32 hi_mpi_isp_get_exposure_attr(hi_vi_pipe vi_pipe, hi_isp_exposure_attr *exp_attr);

hi_s32 hi_mpi_isp_set_wdr_exposure_attr(hi_vi_pipe vi_pipe, const hi_isp_wdr_exposure_attr *wdr_exp_attr);
hi_s32 hi_mpi_isp_get_wdr_exposure_attr(hi_vi_pipe vi_pipe, hi_isp_wdr_exposure_attr *wdr_exp_attr);

hi_s32 hi_mpi_isp_set_ae_route_attr(hi_vi_pipe vi_pipe, const hi_isp_ae_route *ae_route_attr);
hi_s32 hi_mpi_isp_get_ae_route_attr(hi_vi_pipe vi_pipe, hi_isp_ae_route *ae_route_attr);

hi_s32 hi_mpi_isp_set_ae_route_sf_attr(hi_vi_pipe vi_pipe, const hi_isp_ae_route *ae_route_sf_attr);
hi_s32 hi_mpi_isp_get_ae_route_sf_attr(hi_vi_pipe vi_pipe, hi_isp_ae_route *ae_route_sf_attr);

hi_s32 hi_mpi_isp_query_exposure_info(hi_vi_pipe vi_pipe, hi_isp_exp_info *exp_info);

hi_s32 hi_mpi_isp_set_iris_attr(hi_vi_pipe vi_pipe, const hi_isp_iris_attr *iris_attr);
hi_s32 hi_mpi_isp_get_iris_attr(hi_vi_pipe vi_pipe, hi_isp_iris_attr *iris_attr);

hi_s32 hi_mpi_isp_set_dciris_attr(hi_vi_pipe vi_pipe, const hi_isp_dciris_attr *dciris_attr);
hi_s32 hi_mpi_isp_get_dciris_attr(hi_vi_pipe vi_pipe, hi_isp_dciris_attr *dciris_attr);

hi_s32 hi_mpi_isp_set_piris_attr(hi_vi_pipe vi_pipe, const hi_isp_piris_attr *piris_attr);
hi_s32 hi_mpi_isp_get_piris_attr(hi_vi_pipe vi_pipe, hi_isp_piris_attr *piris_attr);

hi_s32 hi_mpi_isp_set_ae_route_attr_ex(hi_vi_pipe vi_pipe, const hi_isp_ae_route_ex *ae_route_attr_ex);
hi_s32 hi_mpi_isp_get_ae_route_attr_ex(hi_vi_pipe vi_pipe, hi_isp_ae_route_ex *ae_route_attr_ex);

hi_s32 hi_mpi_isp_set_ae_route_sf_attr_ex(hi_vi_pipe vi_pipe, const hi_isp_ae_route_ex *ae_route_sf_attr_ex);
hi_s32 hi_mpi_isp_get_ae_route_sf_attr_ex(hi_vi_pipe vi_pipe, hi_isp_ae_route_ex *ae_route_sf_attr_ex);

hi_s32 hi_mpi_isp_set_exp_convert(hi_vi_pipe vi_pipe, hi_isp_exp_conv_param *conv_param);
hi_s32 hi_mpi_isp_get_exp_convert(hi_vi_pipe vi_pipe, hi_isp_exp_conv_param *conv_param);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */

#endif /* __MPI_VI_H__ */
