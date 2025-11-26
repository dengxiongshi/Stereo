/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: Api of 3a
 * Author: Hisilicon multimedia software group
 * Create: 2023-01-05
 */

#ifndef HI_COMMON_3A_H
#define HI_COMMON_3A_H

#include "hi_media_common.h"
#include "hi_common_isp.h"
#include "hi_common_sns.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

#define HI_ISP_ALG_LIB_NAME_SIZE_MAX    20


typedef enum {
    HI_ISP_WDR_MODE_SET = 8000,
    HI_ISP_PROC_WRITE,

    HI_ISP_AE_FPS_BASE_SET,
    HI_ISP_AE_BLC_SET,
    HI_ISP_AE_RC_SET,
    HI_ISP_AE_BAYER_FORMAT_SET,
    HI_ISP_AE_INIT_INFO_GET,

    HI_ISP_AWB_ISO_SET,  /* set iso, change saturation when iso change */
    HI_ISP_CHANGE_IMAGE_MODE_SET,
    HI_ISP_UPDATE_INFO_GET,
    HI_ISP_FRAMEINFO_GET,
    HI_ISP_ATTACHINFO_GET,
    HI_ISP_COLORGAMUTINFO_GET,
    HI_ISP_AWB_INTTIME_SET,
    HI_ISP_BAS_MODE_SET,
    HI_ISP_PROTRIGGER_SET,
    HI_ISP_AWB_PIRIS_SET,
    HI_ISP_AWB_SNAP_MODE_SET,
    HI_ISP_AWB_ZONE_ROW_SET,
    HI_ISP_AWB_ZONE_COL_SET,
    HI_ISP_AWB_ZONE_BIN_SET,
    HI_ISP_AWB_ERR_GET,
    HI_ISP_CTRL_CMD_BUTT,
} hi_isp_ctrl_cmd;

typedef struct {
    hi_char *proc_buff;
    hi_u32   buff_len;
    hi_u32   write_len;   /* the len count should contain '\0'. */
} hi_isp_ctrl_proc_write;

typedef struct {
    hi_bool stitch_enable;
    hi_bool main_pipe;
    hi_u8   stitch_pipe_num;
    hi_s8   stitch_bind_id[HI_VI_MAX_PIPE_NUM];
} hi_isp_stitch_attr;

/* AE */
/* the init param of ae alg */
typedef struct {
    hi_sensor_id sensor_id;
    hi_u8  wdr_mode;
    hi_u8  hdr_mode;
    hi_u16 black_level;
    hi_float fps;
    hi_isp_bayer_format bayer_format;
    hi_isp_stitch_attr stitch_attr;

    hi_s32 rsv;
} hi_isp_ae_param;

/* the statistics of ae alg */
typedef struct {
    hi_u32  pixel_count[HI_ISP_CHN_MAX_NUM];
    hi_u32  pixel_weight[HI_ISP_CHN_MAX_NUM];
    hi_u32  histogram_mem_array[HI_ISP_CHN_MAX_NUM][HI_ISP_HIST_NUM];
} hi_isp_fe_ae_stat_1;

typedef struct {
    hi_u16  global_avg_r[HI_ISP_CHN_MAX_NUM];
    hi_u16  global_avg_gr[HI_ISP_CHN_MAX_NUM];
    hi_u16  global_avg_gb[HI_ISP_CHN_MAX_NUM];
    hi_u16  global_avg_b[HI_ISP_CHN_MAX_NUM];
} hi_isp_fe_ae_stat_2;

typedef struct {
    hi_u16  zone_avg[HI_ISP_CHN_MAX_NUM][HI_ISP_AE_ZONE_ROW][HI_ISP_AE_ZONE_COLUMN][HI_ISP_BAYER_PATTERN_NUM];
} hi_isp_fe_ae_stat_3;

typedef struct {
    hi_u16  zone_avg[HI_VI_MAX_PIPE_NUM][HI_ISP_CHN_MAX_NUM][HI_ISP_AE_ZONE_ROW]
        [HI_ISP_AE_ZONE_COLUMN][HI_ISP_BAYER_PATTERN_NUM];
} hi_isp_fe_ae_stitch_stat_3;

typedef struct {
    hi_u32  pixel_count;
    hi_u32  pixel_weight;
    hi_u32  histogram_mem_array[HI_ISP_HIST_NUM];
} hi_isp_be_ae_stat_1;

typedef struct {
    hi_u16  global_avg_r;
    hi_u16  global_avg_gr;
    hi_u16  global_avg_gb;
    hi_u16  global_avg_b;
} hi_isp_be_ae_stat_2;

typedef struct {
    hi_u16  zone_avg[HI_ISP_AE_ZONE_ROW][HI_ISP_AE_ZONE_COLUMN][HI_ISP_BAYER_PATTERN_NUM];
} hi_isp_be_ae_stat_3;

typedef struct {
    hi_u16  zone_avg[HI_VI_MAX_PIPE_NUM][HI_ISP_AE_ZONE_ROW][HI_ISP_AE_ZONE_COLUMN][HI_ISP_BAYER_PATTERN_NUM];
} hi_isp_be_ae_stitch_stat_3;

typedef struct {
    hi_u32  frame_cnt;    /* the counting of frame */

    hi_isp_fe_ae_stat_1 *fe_ae_stat1;
    hi_isp_fe_ae_stat_2 *fe_ae_stat2;
    hi_isp_fe_ae_stat_3 *fe_ae_stat3;
    hi_isp_fe_ae_stitch_stat_3 *fe_ae_sti_stat;
    hi_isp_be_ae_stat_1 *be_ae_stat1;
    hi_isp_be_ae_stat_2 *be_ae_stat2;
    hi_isp_be_ae_stat_3 *be_ae_stat3;
    hi_isp_be_ae_stitch_stat_3 *be_ae_sti_stat;
} hi_isp_ae_info;

typedef struct {
    hi_bool change;

    hi_bool hist_adjust;
    hi_u8 ae_be_sel;
    hi_u8 four_plane_mode;
    hi_u8 hist_offset_x;
    hi_u8 hist_offset_y;
    hi_u8 hist_skip_x;
    hi_u8 hist_skip_y;

    hi_bool mode_update;
    hi_u8 hist_mode;
    hi_u8 aver_mode;
    hi_u8 max_gain_mode;

    hi_bool wight_table_update;
    hi_u8 weight_table[HI_VI_MAX_PIPE_NUM][HI_ISP_AE_ZONE_ROW][HI_ISP_AE_ZONE_COLUMN];
} hi_isp_ae_stat_attr;

/* the final calculate of ae alg */
typedef struct {
    hi_u32  int_time[4];
    hi_u32  isp_dgain;
    hi_u32  again;
    hi_u32  dgain;
    hi_u32  iso;
    hi_u32  isp_dgain_sf;
    hi_u32  again_sf;
    hi_u32  dgain_sf;
    hi_u32  iso_sf;
    hi_u8   ae_run_interval;

    hi_bool piris_valid;
    hi_s32  piris_pos;
    hi_u32  piris_gain;
    hi_u32  sns_lhcg_exp_ratio;

    hi_isp_fswdr_mode fswdr_mode;
    hi_u32  wdr_gain[HI_ISP_WDR_MAX_FRAME_NUM];
    hi_u32  hmax_times; /* unit is ns */
    hi_u32  vmax; /* unit is line */

    hi_isp_ae_stat_attr stat_attr;
    hi_isp_dcf_update_info update_info;
} hi_isp_ae_result;

typedef struct {
    hi_u32 isp_dgain;
    hi_u32 iso;
} hi_isp_ae_init_info;

typedef struct {
    hi_s32 (*pfn_ae_init)(hi_s32 handle, const hi_isp_ae_param *ae_param);
    hi_s32 (*pfn_ae_run)(hi_s32 handle,
                         const hi_isp_ae_info *ae_info,
                         hi_isp_ae_result *ae_result,
                         hi_s32 rsv);
    hi_s32 (*pfn_ae_ctrl)(hi_s32 handle, hi_u32 cmd, hi_void *value);
    hi_s32 (*pfn_ae_exit)(hi_s32 handle);
} hi_isp_ae_exp_func;

typedef struct {
    hi_isp_ae_exp_func ae_exp_func;
} hi_isp_ae_register;

/* the init param of awb alg */
typedef struct {
    hi_sensor_id sensor_id;
    hi_u8 wdr_mode;
    hi_u8 awb_zone_row;
    hi_u8 awb_zone_col;
    hi_u8 awb_zone_bin;
    hi_isp_stitch_attr stitch_attr;
    hi_u16 awb_width;
    hi_u16 awb_height;
    hi_u32 init_iso;
    hi_s8 rsv;
} hi_isp_awb_param;

/* the statistics of awb alg */
typedef struct {
    hi_u16  metering_awb_avg_r;
    hi_u16  metering_awb_avg_g;
    hi_u16  metering_awb_avg_b;
    hi_u16  metering_awb_count_all;
} hi_isp_awb_stat_1;

typedef struct {
    hi_u16 *zone_avg_r;
    hi_u16 *zone_avg_g;
    hi_u16 *zone_avg_b;
    hi_u16 *zone_count;
} hi_isp_awb_stat_result;

typedef struct {
    hi_u32  frame_cnt;

    hi_isp_awb_stat_1 *awb_stat1;
    hi_isp_awb_stat_result awb_stat2;
    hi_u8  awb_gain_switch;
    hi_u8  awb_stat_switch;
    hi_bool wb_gain_in_sensor;
    hi_u32 wdr_wb_gain[HI_ISP_BAYER_CHN_NUM];
} hi_isp_awb_info;

/* the statistics's attr of awb alg */
typedef struct {
    hi_bool stat_cfg_update;

    hi_u16  metering_white_level_awb;
    hi_u16  metering_black_level_awb;
    hi_u16  metering_cr_ref_max_awb;
    hi_u16  metering_cb_ref_max_awb;
    hi_u16  metering_cr_ref_min_awb;
    hi_u16  metering_cb_ref_min_awb;
} hi_isp_awb_raw_stat_attr;

/* the final calculate of awb alg */
typedef struct {
    hi_u32  white_balance_gain[HI_ISP_BAYER_CHN_NUM];
    hi_u16  color_matrix[HI_ISP_CCM_MATRIX_SIZE];
    hi_u32  color_temp;
    hi_u8   saturation;
    hi_isp_awb_raw_stat_attr raw_stat_attr;
} hi_isp_awb_result;

typedef struct {
    hi_s32 (*pfn_awb_init)(hi_s32 handle, const hi_isp_awb_param *awb_param, hi_isp_awb_result *awb_result);
    hi_s32 (*pfn_awb_run)(hi_s32 handle,
                          const hi_isp_awb_info *awb_info,
                          hi_isp_awb_result *awb_result,
                          hi_s32 rsv);
    hi_s32 (*pfn_awb_ctrl)(hi_s32 handle, hi_u32 cmd, hi_void *value);
    hi_s32 (*pfn_awb_exit)(hi_s32 handle);
} hi_isp_awb_exp_func;

typedef struct {
    hi_isp_awb_exp_func awb_exp_func;
} hi_isp_awb_register;

/* AF */

/* the init param of af alg */
typedef struct {
    hi_sensor_id sensor_id;
    hi_u8 wdr_mode;
    hi_u8 af_zone_row;
    hi_u8 af_zone_col;
    hi_s32 s32Rsv;
} hi_isp_af_param;

/* the statistics of af alg */
typedef struct {
    hi_u16  v1;
    hi_u16  h1;
    hi_u16  v2;
    hi_u16  h2;
    hi_u16  y;
    hi_u16  hl_cnt;
} hi_isp_af_zone;

typedef struct {
    /* R; the zoned measure of contrast */
    hi_isp_af_zone zone_metrics[HI_ISP_WDR_CHN_MAX][HI_ISP_AF_ZONE_ROW][HI_ISP_AF_ZONE_COLUMN];
} hi_isp_fe_af_stat;

typedef struct {
    hi_isp_af_zone zone_metrics[HI_ISP_AF_ZONE_ROW][HI_ISP_AF_ZONE_COLUMN]; /* R; the zoned measure of contrast */
} hi_isp_be_af_stat;

typedef struct {
    hi_u32 frame_cnt;
    hi_isp_be_af_stat *af_stat;
}hi_isp_af_info;

/* the final calculate of af alg */
typedef struct {
    hi_s32 s32Rsv;
} hi_isp_af_result;

typedef struct {
    hi_s32 (*pfn_af_init)(hi_s32 s32Handle, const hi_isp_af_param *af_param);
    hi_s32 (*pfn_af_run)(hi_s32 s32Handle,
        const hi_isp_af_info *af_info,
        hi_isp_af_result *af_result,
        hi_s32 s32Rsv
        );
    hi_s32 (*pfn_af_ctrl)(hi_s32 s32Handle, hi_u32 u32Cmd, hi_void *pValue);
    hi_s32 (*pfn_af_exit)(hi_s32 s32Handle);
} hi_isp_af_exp_func;

typedef struct {
    hi_isp_af_exp_func af_exp_func;
}hi_isp_af_register;

typedef struct {
    hi_s32  id;
    hi_char lib_name[HI_ISP_ALG_LIB_NAME_SIZE_MAX];
} hi_isp_3a_alg_lib;

typedef struct {
    hi_sensor_id       sensor_id;
    hi_isp_3a_alg_lib  ae_lib;
    hi_isp_3a_alg_lib  af_lib;
    hi_isp_3a_alg_lib  awb_lib;
} hi_isp_bind_attr;

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */

#endif /* __HI_COMM_SNS_H__ */
