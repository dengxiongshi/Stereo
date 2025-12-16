/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: Api of awb
 * Author: Hisilicon multimedia software group
 * Create: 2023-01-05
 */

#ifndef HI_COMMON_AWB_H
#define HI_COMMON_AWB_H

#define HI_AWB_LIB_NAME "hisi_awb_lib"

#include "hi_media_type.h"
#include "hi_common_isp.h"
#include "hi_common_3a.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* end of #ifdef __cplusplus */

/* sensor's interface to awb */
typedef struct {
    hi_u16 color_temp;              /* RW;  range:[2000,10000]; format:16.0; the current color temperature */
    hi_u16 ccm[HI_ISP_CCM_MATRIX_SIZE];    /* RW;  range: [0x0, 0xFFFF]; format:8.8;
                                       CCM matrix for different color temperature */
} hi_isp_awb_ccm_tab;

typedef struct {
    hi_u16  ccm_tab_num;                   /* RW;  range: [0x3, 0x7]; format:16.0; the number of CCM matrix */
    hi_isp_awb_ccm_tab ccm_tab[HI_ISP_CCM_MATRIX_NUM];
} hi_isp_awb_ccm;

typedef struct {
    hi_bool valid;

    hi_u8   saturation[HI_ISP_AUTO_ISO_NUM];   /* RW;adjust saturation, different iso with different saturation */
} hi_isp_awb_agc_table;

typedef struct {
    hi_u16  wb_ref_temp;       /* RW;reference color temperature for WB  */
    hi_u16  gain_offset[HI_ISP_BAYER_CHN_NUM];  /* RW; gain offset for white balance */
    hi_s32  wb_para[HI_ISP_AWB_CURVE_PARA_NUM];      /* RW; parameter for wb curve,p1,p2,q1,a1,b1,c1 */

    hi_u16  golden_rgain;      /* rgain for the golden sample */
    hi_u16  golden_bgain;      /* bgain for the golden sample */
    hi_u16  sample_rgain;      /* rgain for the current sample */
    hi_u16  sample_bgain;      /* bgain for the current sample */
    hi_isp_awb_agc_table agc_tbl;
    hi_isp_awb_ccm ccm;
    hi_u16    init_rgain;           /* init WB gain */
    hi_u16    init_ggain;
    hi_u16    init_bgain;
    hi_u8     awb_run_interval;       /* RW;AWB run interval */
    hi_u16    init_ccm[HI_ISP_CCM_MATRIX_SIZE];
} hi_isp_awb_sensor_default;

typedef struct {
    hi_s32 (*pfn_cmos_get_awb_default)(hi_vi_pipe vi_pipe, hi_isp_awb_sensor_default *awb_sns_dft);
} hi_isp_awb_sensor_exp_func;

typedef struct {
    hi_isp_awb_sensor_exp_func sns_exp;
} hi_isp_awb_sensor_register;

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* end of #ifdef __cplusplus */

#endif
