/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: Api of ae
 * Author: Hisilicon multimedia software group
 * Create: 2023-01-05
 */

#ifndef HI_COMMON_AE_H
#define HI_COMMON_AE_H

#include "hi_media_type.h"
#include "hi_common_isp.h"
#include "hi_common_3a.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

#define HI_AE_LIB_NAME "hisi_ae_lib"

/* ae ctrl cmd */
typedef enum {
    HI_ISP_AE_DEBUG_ATTR_SET,
    HI_ISP_AE_DEBUG_ATTR_GET,

    HI_ISP_AE_CTRL_BUTT,
} hi_isp_ae_ctrl_cmd;

typedef enum {
    HI_ISP_AE_ACCURACY_DB = 0,
    HI_ISP_AE_ACCURACY_LINEAR,
    HI_ISP_AE_ACCURACY_TABLE,

    HI_ISP_AE_ACCURACY_BUTT,
} hi_isp_ae_accuracy_type;

typedef struct {
    hi_isp_ae_accuracy_type accu_type;
    float   accuracy;
    float   offset;
} hi_isp_ae_accuracy;

typedef struct {
    hi_bool quick_start_enable;
    hi_u8 black_frame_num;
    hi_bool ir_mode_en;
    hi_u32 init_exposure_ir;
    hi_u32 iso_thr_ir;
    hi_u16 ir_cut_delay_time;
} hi_isp_quick_start_param;

typedef struct {
    hi_u8   hist_thresh[HI_ISP_HIST_THRESH_NUM];
    hi_u8   ae_compensation;

    hi_u32  lines_per500ms;
    hi_u32  flicker_freq;
    hi_float fps;
    hi_u32  hmax_times; /* unit is ns */
    hi_u32  init_exposure;
    hi_u32  init_ae_speed;
    hi_u32  init_ae_tolerance;

    hi_u32  full_lines_std;
    hi_u32  full_lines_max;
    hi_u32  full_lines;
    hi_u32  binning_full_lines;
    hi_u32  max_int_time;     /* RW;unit is line */
    hi_u32  min_int_time;
    hi_u32  max_int_time_target;
    hi_u32  min_int_time_target;
    hi_isp_ae_accuracy int_time_accu;

    hi_u32  max_again;
    hi_u32  min_again;
    hi_u32  max_again_target;
    hi_u32  min_again_target;
    hi_isp_ae_accuracy again_accu;

    hi_u32  max_dgain;
    hi_u32  min_dgain;
    hi_u32  max_dgain_target;
    hi_u32  min_dgain_target;
    hi_isp_ae_accuracy dgain_accu;

    hi_u32  max_isp_dgain_target;
    hi_u32  min_isp_dgain_target;
    hi_u32  isp_dgain_shift;

    hi_u32  max_int_time_step;
    hi_bool max_time_step_enable;
    hi_u32  max_inc_time_step[HI_ISP_WDR_MAX_FRAME_NUM];
    hi_u32  max_dec_time_step[HI_ISP_WDR_MAX_FRAME_NUM];
    hi_u32  lf_max_short_time;
    hi_u32  lf_min_exposure;

    hi_isp_ae_route ae_route_attr;
    hi_bool ae_route_ex_valid;
    hi_isp_ae_route_ex ae_route_attr_ex;

    hi_isp_ae_route ae_route_sf_attr;
    hi_isp_ae_route_ex ae_route_sf_attr_ex;

    hi_u16 man_ratio_enable;
    hi_u32 arr_ratio[HI_ISP_EXP_RATIO_NUM];
    hi_isp_iris_type  iris_type;
    hi_isp_piris_attr piris_attr;
    hi_isp_iris_f_no  max_iris_fno;  /* RW; Range:[0, 10]; Format:4.0;
                                        Max F number of Piris's aperture, it's related to the specific iris */
    hi_isp_iris_f_no  min_iris_fno;  /* RW; Range:[0, 10]; Format:4.0;
                                        Min F number of Piris's aperture, it's related to the specific iris */
    hi_isp_ae_strategy ae_exp_mode;

    hi_u16 iso_cal_coef;
    hi_u8  ae_run_interval;
    hi_u32 exp_ratio_max;
    hi_u32 exp_ratio_min;
    hi_bool diff_gain_support;
    hi_isp_quick_start_param quick_start;
    hi_isp_prior_frame prior_frame;
    hi_bool ae_gain_sep_cfg;
    hi_bool lhcg_support;
    hi_u32 sns_lhcg_exp_ratio;
} hi_isp_ae_sensor_default;

typedef struct {
    hi_isp_fswdr_mode fswdr_mode;
} hi_isp_ae_fswdr_attr;

typedef struct {
    hi_u32 reg_addr;
    hi_u32 reg_value;
} hi_isp_ae_param_reg;

typedef struct {
    hi_u32 tar_fps;
    hi_u32 exp_time;
    hi_u32 exp_again;
    hi_u32 exp_dgain;
    hi_u32 exp_isp_dgain;
    hi_isp_ae_param_reg time_reg[10]; /* 10 */
    hi_isp_ae_param_reg again_reg[10]; /* 10 */
    hi_isp_ae_param_reg dgain_reg[10]; /* 10 */
} hi_isp_ae_convert_param;

typedef struct {
    hi_s32 (*pfn_cmos_get_ae_default)(hi_vi_pipe vi_pipe, hi_isp_ae_sensor_default *ae_sns_dft);
    /* the function of sensor set fps */
    hi_void (*pfn_cmos_fps_set)(hi_vi_pipe vi_pipe, hi_float f32_fps, hi_isp_ae_sensor_default *ae_sns_dft);
    hi_void (*pfn_cmos_slow_framerate_set)(hi_vi_pipe vi_pipe, hi_u32 full_lines, hi_isp_ae_sensor_default *ae_sns_dft);

    /* while isp notify ae to update sensor regs, ae call these funcs. */
    hi_void (*pfn_cmos_inttime_update)(hi_vi_pipe vi_pipe, hi_u32 int_time);
    hi_void (*pfn_cmos_gains_update)(hi_vi_pipe vi_pipe, hi_u32 again, hi_u32 dgain);

    hi_void (*pfn_cmos_again_calc_table)(hi_vi_pipe vi_pipe, hi_u32 *again_lin, hi_u32 *again_db);
    hi_void (*pfn_cmos_dgain_calc_table)(hi_vi_pipe vi_pipe, hi_u32 *dgain_lin, hi_u32 *dgain_db);

    hi_void (*pfn_cmos_get_inttime_max)(hi_vi_pipe vi_pipe, hi_u16 man_ratio_enable,
        hi_u32 *ratio, hi_u32 *int_time_max, hi_u32 *int_time_min, hi_u32 *lf_max_int_time);

    /* long frame mode set */
    hi_void (*pfn_cmos_ae_fswdr_attr_set)(hi_vi_pipe vi_pipe, hi_isp_ae_fswdr_attr *ae_fswdr_attr);
    hi_void (*pfn_cmos_ae_quick_start_status_set)(hi_vi_pipe vi_pipe, hi_bool quick_start_status);
    hi_void (*pfn_cmos_exp_param_convert)(hi_vi_pipe vi_pipe, hi_isp_ae_convert_param *exp_param);
} hi_isp_ae_sensor_exp_func;

typedef struct {
    hi_isp_ae_sensor_exp_func sns_exp;
} hi_isp_ae_sensor_register;

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */

#endif
