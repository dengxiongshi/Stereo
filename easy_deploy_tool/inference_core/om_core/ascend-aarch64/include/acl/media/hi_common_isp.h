/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: Head of isp
 * Author: Hisilicon multimedia software group
 * Create: 2023-01-05
 */
#ifndef MPI_COMM_ISP_H
#define MPI_COMM_ISP_H

#include "hi_common_vi.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

/* isp sys part */
#define HI_ISP_STRIPING_MAX_NUM         5

#define HI_ISP_MAX_STITCH_NUM           1

#define HI_ISP_MAX_PHY_PIPE_NUM         HI_VI_MAX_PHY_PIPE_NUM
#define HI_ISP_MAX_VIR_PIPE_NUM         HI_VI_MAX_VIR_PIPE_NUM
#define HI_ISP_MAX_PIPE_NUM             (HI_ISP_MAX_PHY_PIPE_NUM + HI_ISP_MAX_VIR_PIPE_NUM)

/* isp alg part */
#define HI_ISP_CLUT_LUT_LENGTH          5508
#define HI_ISP_CA_YRATIO_LUT_LENGTH     128

#define HI_ISP_STATIC_DP_COUNT_NORMAL   2048
#define HI_ISP_STATIC_DP_COUNT_MAX      (HI_ISP_STATIC_DP_COUNT_NORMAL * HI_ISP_STRIPING_MAX_NUM)

#define HI_ISP_AWB_ZONE_BIN             1
#define HI_ISP_EXPANDER_NODE_NUM        257

#define HI_ISP_LSC_GRID_COL             33
#define HI_ISP_LSC_GRID_ROW             33
#define HI_ISP_LSC_GRID_POINTS          (HI_ISP_LSC_GRID_COL * HI_ISP_LSC_GRID_ROW)
#define HI_ISP_LSC_DEFAULT_MESH_SCALE   3

#define hi_isp_lsc_xgrid_width_max(x)   ((x) / 4 - 60)
#define hi_isp_lsc_ygrid_width_max(x)   ((x) / 4 - 60)

#define HI_ISP_CCM_MATRIX_SIZE          9
#define HI_ISP_CCM_MATRIX_NUM           7

/* MACRO DEFINITION */
#define HI_CFACOLORPLANE                3

#define HI_ISP_AWB_ZONE_ORIG_ROW        32
#define HI_ISP_AWB_ZONE_ORIG_COLUMN     32
#define HI_ISP_AWB_ZONE_BIN_MAX         4
#define HI_ISP_AE_ZONE_ROW              15
#define HI_ISP_AE_ZONE_COLUMN           17
#define HI_ISP_MG_ZONE_ROW              15
#define HI_ISP_MG_ZONE_COLUMN           17
#define HI_ISP_AWB_ZONE_NUM             (HI_ISP_AWB_ZONE_ORIG_ROW * HI_ISP_AWB_ZONE_ORIG_COLUMN * HI_ISP_AWB_ZONE_BIN)
#define HI_ISP_AWB_ZONE_STITCH_MAX      (HI_ISP_AWB_ZONE_NUM * HI_ISP_MAX_STITCH_NUM)
#define HI_ISP_AF_ZONE_ROW              15
#define HI_ISP_AF_ZONE_COLUMN           17

#define HI_ISP_GAMMA_NODE_NUM           1025
#define HI_ISP_PREGAMMA_NODE_NUM        257

#define HI_ISP_AUTO_ISO_NUM             16
#define HI_ISP_SHARPEN_LUMA_NUM         32
#define HI_ISP_SHARPEN_GAIN_NUM         32

#define HI_ISP_MAX_SNS_REGS             32

#define HI_ISP_RLSC_POINTS                129
#define HI_ISP_MLSC_X_HALF_GRID_NUM       ((HI_ISP_LSC_GRID_COL - 1) / 2)
#define HI_ISP_MLSC_Y_HALF_GRID_NUM       ((HI_ISP_LSC_GRID_ROW - 1) / 2)
#define HI_ISP_MLSC_GAIN_LUT_NUM          2
#define HI_ISP_ACS_LIGHT_NUM              32
#define HI_ISP_ACS_CHN_NUM                2

#define HI_ISP_WDR_MAX_FRAME_NUM          4

#define HI_ISP_CSC_DC_NUM                 3
#define HI_ISP_CSC_COEF_NUM               9

#define HI_ISP_BAYER_CALIBTAION_MAX_NUM            50
#define HI_ISP_BAYERNR_CORINGLOW_STRENGTH_DIVISOR  10000
#define HI_ISP_AI_MAX_STEP_FNO_NUM                 1024

#define HI_ISP_BAYERNR_LUT_LENGTH       33
#define HI_ISP_RGB_CHN_NUM              3

#define HI_ISP_BAYER_CHN_NUM            4

#define HI_ISP_NOISESET_ELENUM          7

#define HI_ISP_DRC_CC_NODE_NUM          33
#define HI_ISP_DRC_TM_NODE_NUM          200

#define HI_ISP_DEHAZE_LUT_SIZE          256

#define HI_ISP_DE_LUMA_GAIN_LUT_N       17
#define HI_ISP_EXP_RATIO_STRENGTH_NUM   16
#define HI_ISP_LCAC_EXP_RATIO_NUM       16
#define HI_ISP_LCAC_DET_NUM             3

#define HI_ISP_DO_NOT_NEED_SWITCH_IMAGEMODE (-2)

#define HI_ISP_HIST_THRESH_NUM          4

#define ccm_convert_pre(value)  (((value) & 0x8000U) | ((value) << 2))
#define ccm_convert(value)      (((value) & 0x8000U) ? ((~((value) & 0x7FFFU)) + 1U) : (value))

/* GENERAL STRUCTURES */
/*
 * ISP Error Code
 * 0x40 : ISP_NOT_INIT
 * 0x41 : ISP_MEM_NOT_INIT
 * 0x42 : ISP_ATTR_NOT_CFG
 * 0x43 : ISP_SNS_UNREGISTER
 * 0x44 : ISP_INVALID_ADDR
 * 0x45 : ISP_NOMEM
 * 0x46 : ISP_NO_INT
 */
typedef enum {
    HI_ERR_ISP_NOT_INIT                = 0x40, /* ISP not init */
    HI_ERR_ISP_MEM_NOT_INIT            = 0x41, /* ISP memory not init */
    HI_ERR_ISP_ATTR_NOT_CFG            = 0x42, /* ISP attribute not cfg */
    HI_ERR_ISP_SNS_UNREGISTER          = 0x43, /* ISP sensor unregister */
    HI_ERR_ISP_INVALID_ADDR            = 0x44, /* ISP invalid address */
    HI_ERR_ISP_NOMEM                   = 0x45, /* ISP nomem */
    HI_ERR_ISP_NO_INT                  = 0x46, /* ISP */
} hi_isp_err_code;

#define HI_ERR_ISP_NULL_PTR         HI_DEFINE_ERR(HI_ID_ISP, HI_ERR_LEVEL_ERROR, HI_ERR_NULL_PTR)
#define HI_ERR_ISP_ILLEGAL_PARAM    HI_DEFINE_ERR(HI_ID_ISP, HI_ERR_LEVEL_ERROR, HI_ERR_ILLEGAL_PARAM)
#define HI_ERR_ISP_NOT_SUPPORT      HI_DEFINE_ERR(HI_ID_ISP, HI_ERR_LEVEL_ERROR, HI_ERR_NOT_SUPPORT)
#define HI_ERR_ISP_NO_MEM           HI_DEFINE_ERR(HI_ID_ISP, HI_ERR_LEVEL_ERROR, HI_ERR_NO_MEM)

#define HI_ERR_ISP_NOT_INIT         HI_DEFINE_ERR(HI_ID_ISP, HI_ERR_LEVEL_ERROR, HI_ERR_ISP_NOT_INIT)
#define HI_ERR_ISP_MEM_NOT_INIT     HI_DEFINE_ERR(HI_ID_ISP, HI_ERR_LEVEL_ERROR, HI_ERR_ISP_MEM_NOT_INIT)
#define HI_ERR_ISP_ATTR_NOT_CFG     HI_DEFINE_ERR(HI_ID_ISP, HI_ERR_LEVEL_ERROR, HI_ERR_ISP_ATTR_NOT_CFG)
#define HI_ERR_ISP_SNS_UNREGISTER   HI_DEFINE_ERR(HI_ID_ISP, HI_ERR_LEVEL_ERROR, HI_ERR_ISP_SNS_UNREGISTER)
#define HI_ERR_ISP_INVALID_ADDR     HI_DEFINE_ERR(HI_ID_ISP, HI_ERR_LEVEL_ERROR, HI_ERR_ISP_INVALID_ADDR)
#define HI_ERR_ISP_NO_INT           HI_DEFINE_ERR(HI_ID_ISP, HI_ERR_LEVEL_ERROR, HI_ERR_ISP_NO_INT)

/* Defines the structure of ISP control parameters. */
typedef struct {
    hi_u8   be_buf_num;      /* RW;Range:[2, 9]; offline be cfg buffer number(Only used for offline mode).
                                Not support changed after isp init */
    hi_u32  proc_param;      /* RW;Format:32.0; Indicate the update frequency of ISP_PROC information,
                                No distinction vipipe */
    hi_u32  stat_interval;      /* RW;Range:(0, 0xffffffff];Format:32.0;
                                Indicate the time interval of ISP statistic information */
    hi_u32  update_pos;      /* RW;Range:[0, 1]; Format:32.0; Indicate the location of the configuration register of
                                ISP interrupt */
    hi_u32  interrupt_time_out;    /* RW;Format:32.0; Indicate the time(unit:ms) of interrupt timeout */
    hi_u32  pwm_num;      /* R;Format:32.0; Indicate PWM number */
    hi_u32  port_interrupt_delay;  /* RW;Format:32.0; Port interrupt delay value, unit:clk */
    hi_bool ldci_tpr_flt_en; /* R;Indicate Enable LDCI temporal filter. Not support changed after isp init */
} hi_isp_ctrl_param;

/*
 * Defines the prior frame of ISP
 * 0 = long frame
 * 1 = short frame
 */
typedef enum {
    HI_ISP_LONG_FRAME      = 0,
    HI_ISP_SHORT_FRAME       = 1,
    HI_ISP_PRIOR_FRAME_BUTT
} hi_isp_prior_frame;

/* Defines the working mode of ISP : 0 = automatic mode, 1 = manual mode */
typedef enum {
    HI_ISP_OP_TYPE_AUTO    = 0,
    HI_ISP_OP_TYPE_MANUAL  = 1,
    HI_ISP_OP_TYPE_BUTT
} hi_isp_op_type;

/*
 * Defines the ISP correction or detection status
 * 0 = initial status, no calibration
 * 1 = The static defect pixel calibration ends normally
 * 2 = The static defect pixel calibration ends due to timeout.
 */
typedef enum {
    HI_ISP_STATE_INIT     = 0,
    HI_ISP_STATE_SUCCESS  = 1,
    HI_ISP_STATE_TIMEOUT  = 2,
    HI_ISP_STATE_BUTT
} hi_isp_status;

/* Defines the format of the input Bayer image */
typedef enum {
    HI_ISP_BAYER_RGGB    = 0,
    HI_ISP_BAYER_GRBG    = 1,
    HI_ISP_BAYER_GBRG    = 2,
    HI_ISP_BAYER_BGGR    = 3,
    HI_ISP_BAYER_BUTT
} hi_isp_bayer_format;

/* Defines the bitwidth of the input Bayer image, used for lsc online calibration */
typedef enum {
    HI_ISP_BAYER_RAWBIT_8BIT    = 8,
    HI_ISP_BAYER_RAWBIT_10BIT   = 10,
    HI_ISP_BAYER_RAWBIT_12BIT   = 12,
    HI_ISP_BAYER_RAWBIT_14BIT   = 14,
    HI_ISP_BAYER_RAWBIT_16BIT   = 16,
    HI_ISP_BAYER_RAWBIT_BUTT
} hi_isp_bayer_raw_bit;

/* ISP public attribute, contains the public image attribute */
typedef struct {
    hi_rect             wnd_rect;        /* RW; start position of the cropping window, image width, and image height */
    hi_size             sns_size;        /* RW; width and height of the image output from the sensor */
    hi_float            frame_rate;      /* RW; range: [0, 0xFFFF]; for frame rate */
    hi_isp_bayer_format bayer_format;    /* RW; range:[0,3];format:2.0;the format of the input bayer image */
    hi_wdr_mode         wdr_mode;        /* RW; WDR mode select */
    hi_u8               sns_mode;        /* RW; range: [0, 0xFF]; format: 8.0; for special sensor mode switches */
} hi_isp_pub_attr;

/* Slave mode sensor sync signal generate module */
typedef struct {
    union {
        struct {
            hi_u32  bit16_rsv           :  16;
            hi_u32  bit_h_inv           :  1;
            hi_u32  bit_v_inv           :  1;
            hi_u32  bit12_rsv           :  12;
            hi_u32  bit_h_enable        :  1;
            hi_u32  bit_v_enable        :  1;
        } bits;
        hi_u32 bytes;
    } cfg;

    hi_u32  vs_time;
    hi_u32  hs_time;
    hi_u32  vs_cyc;
    hi_u32  hs_cyc;
    hi_u32  hs_dly_cyc;
    hi_u32  slave_mode_time;
} hi_isp_slave_sns_sync;

/* Defines the ISP stitch attribute. */
/*
 * Defines the ISP firmware status
 * 0 = Running status
 * 1 = Frozen status
 */
typedef enum {
    HI_ISP_FMW_STATE_RUN = 0,
    HI_ISP_FMW_STATE_FREEZE,
    HI_ISP_FMW_STATE_BUTT
} hi_isp_fmw_state;

/* Defines the WDR mode of the ISP */
typedef struct {
    hi_wdr_mode  wdr_mode;
} hi_isp_wdr_mode;

typedef union {
    hi_u64  key;
    struct {
        hi_u64  bit_bypass_isp_d_gain        : 1;   /* RW;[0] */
        hi_u64  bit_bypass_anti_false_color  : 1;   /* RW;[1] */
        hi_u64  bit_bypass_crosstalk_removal : 1;   /* RW;[2] */
        hi_u64  bit_bypass_dpc            : 1;   /* RW;[3] */
        hi_u64  bit_bypass_nr             : 1;   /* RW;[4] */
        hi_u64  bit_bypass_dehaze         : 1;   /* RW;[5] */
        hi_u64  bit_bypass_wb_gain        : 1;   /* RW;[6] */
        hi_u64  bit_bypass_mesh_shading   : 1;   /* RW;[7] */
        hi_u64  bit_bypass_drc            : 1;   /* RW;[8] */
        hi_u64  bit_bypass_demosaic       : 1;   /* RW;[9] */
        hi_u64  bit_bypass_color_matrix   : 1;   /* RW;[10] */
        hi_u64  bit_bypass_gamma          : 1;   /* RW;[11] */
        hi_u64  bit_bypass_fswdr          : 1;   /* RW;[12] */
        hi_u64  bit_bypass_ca             : 1;   /* RW;[13] */
        hi_u64  bit_bypass_csc            : 1;   /* RW;[14] */
        hi_u64  bit_bypass_radial_crop    : 1;   /* RW;[15]; Not support */
        hi_u64  bit_bypass_sharpen        : 1;   /* RW;[16] */
        hi_u64  bit_bypass_local_cac      : 1;   /* RW;[17] */
        hi_u64  bit_bypass_global_cac     : 1;   /* RW;[18] */
        hi_u64  bit2_chn_select           : 2;   /* RW;[19:20] */
        hi_u64  bit_bypass_ldci           : 1;   /* RW;[21] */
        hi_u64  bit_bypass_pregamma       : 1;   /* RW;[22] */
        hi_u64  bit_bypass_radial_shading : 1;   /* RW;[23]; Not support */
        hi_u64  bit_bypass_ae_stat_fe     : 1;   /* RW;[24] */
        hi_u64  bit_bypass_ae_stat_be     : 1;   /* RW;[25] */
        hi_u64  bit_bypass_mg_stat        : 1;   /* RW;[26] */
        hi_u64  bit_bypass_de             : 1;   /* RW;[27] */
        hi_u64  bit_bypass_af_stat_fe     : 1;   /* RW;[28]; Not support */
        hi_u64  bit_bypass_af_stat_be     : 1;   /* RW;[29] */
        hi_u64  bit_bypass_awb_stat       : 1;   /* RW;[30] */
        hi_u64  bit_bypass_clut           : 1;   /* RW;[31] */
        hi_u64  bit_bypass_hlc            : 1;   /* RW;[32] */
        hi_u64  bit_bypass_edge_mark      : 1;   /* RW;[33] */
        hi_u64  bit_bypass_rgbir          : 1;   /* RW;[34] */
        hi_u64  bit_reserved29            : 29;  /* H; [35:63] */
    };
} hi_isp_module_ctrl;

/*
 * Defines the ISP WDR merge mode.
 * 0 = WDR mode is used
 * 1 = Fusion mode is used
 */
typedef enum {
    HI_ISP_MERGE_WDR_MODE      = 0,
    HI_ISP_MERGE_FUSION_MODE   = 1,
    HI_ISP_MERGE_BUTT
} hi_isp_wdr_merge_mode;

/*
 * 0 = The function of bnr is off
 * 1 = The function of bnr is on
 */
typedef enum {
    HI_ISP_BNR_OFF_MODE         = 0,
    HI_ISP_BNR_ON_MODE          = 1,
    HI_ISP_BNR_BUTT
} hi_isp_bnr_mode;

typedef enum {
    HI_ISP_WDR_WBGAIN_IN_DG1         = 0,
    HI_ISP_WDR_WBGAIN_IN_WB          = 1,
    HI_ISP_WDR_WBGAIN_POSITION_BUTT
} hi_isp_wdr_wb_gain_position;

typedef struct {
    hi_u8 md_thr_low_gain;
    hi_u8 md_thr_hig_gain;
} hi_isp_fswdr_manual_attr;

typedef struct {
    hi_u8 md_thr_low_gain[HI_ISP_AUTO_ISO_NUM];
    hi_u8 md_thr_hig_gain[HI_ISP_AUTO_ISO_NUM];
} hi_isp_fswdr_auto_attr;

typedef struct {
    hi_bool short_expo_chk;     /* RW;Range:[0x0,0x1];Format:1.0; */
    hi_u16  short_check_thd;   /* RW;Range:[0x0,0xFFF];Format:12.0 */
    hi_bool md_ref_flicker;
    hi_u8   mdt_still_thd;     /* RW;Range:[0x0,0xFE];Format:8.0 */
    hi_u8   mdt_full_thd;      /* RW;Range:[0x0,0xFE];Format:8.0;Not support */
    hi_u8   mdt_long_blend;    /* RW;Range:[0x0,0xFE] */
    hi_op_mode op_type;
    hi_isp_fswdr_manual_attr manual_attr;
    hi_isp_fswdr_auto_attr   auto_attr;
} hi_isp_fswdr_mdt_attr;

typedef struct {
    hi_bool motion_comp;        /* RW;Range:[0, 0x1];Format:1.0; enable/disable motion compensation */
    hi_u16  short_thr;          /* RW;Range:[0, 0xFFF];Format:12.0; data above this threshold will be taken from
                                   short exposure only. */
    hi_u16  long_thr;           /* RW;Range:[0, 0xFFF];Format:12.0; limited range:[0x0, short_thr], data below this
                                   threshold will be taken from long exposure only. */
    hi_bool force_long;         /* RW;Range:[0, 1];Format:1.0; enable/disable force long */
    hi_u16  force_long_low_thr; /* RW;Range:[0, 0xFFF];Format:12.0; data above this threshold will force to choose
                                   long frame only */
    hi_u16  force_long_hig_thr; /* RW;Range:[0, 0xFFF];Format:12.0; data below this threshold will force to choose
                                   long frame only */
    hi_isp_fswdr_mdt_attr wdr_mdt;
} hi_isp_wdr_combine_attr;

typedef struct {
    hi_u16  fusion_thr[HI_ISP_WDR_MAX_FRAME_NUM]; /* RW;Range:[0x0,0x3FFF];Format:14.0;the threshold of the 4 frame */
} hi_isp_fusion_attr;

typedef struct {
    hi_isp_wdr_merge_mode   wdr_merge_mode;
    hi_isp_wdr_combine_attr wdr_combine;
    hi_isp_fusion_attr      fusion_attr;
} hi_isp_wdr_fs_attr;

typedef struct {
    hi_u8   asymmetry;    /* RW; Range:[0x1, 0x1E]; Format:5.0; The parameter0 of DRC asymmetry tone mapping curve */
    hi_u8   second_pole;  /* RW; Range:[0x96,0xD2]; Format:8.0; The parameter1 of DRC asymmetry tone mapping curve */
    hi_u8   stretch;      /* RW; Range:[0x1E,0x3C]; Format:6.0; The parameter2 of DRC asymmetry tone mapping curve */
    hi_u8   compress;     /* RW; Range:[0x64,0xC8]; Format:8.0; The parameter3 of DRC asymmetry tone mapping curve */
} hi_isp_drc_asymmetry_curve_attr;

typedef struct {
    hi_u16 strength; /* RW; Range: [0x0, 0x3FF]
                        Strength of dynamic range compression. Higher values lead to higher differential gain between
                        shadows and highlights. */
} hi_isp_drc_manual_attr;

typedef struct {
    hi_u16 strength;  /* RW; Range: [0x0, 0x3FF]
                         It is the base strength. The strength used in ISP is generated by firmware.
                         n linear mode, strength = f1(u16Strength, histogram)
                         In sensor WDR mode: strength = f2(u16Strength, histogram)
                         In 2to1 WDR mode: strength = f3(ExpRatio) */

    hi_u16 strength_max; /* RW; Range: [0x0, 0x3FF]; Maximum DRC strength in Auto mode */
    hi_u16 strength_min; /* RW; [0x0, 0x3FF]; Minimum DRC strength in Auto mode */
} hi_isp_drc_auto_attr;

/* DRC curve type: 0 = Asymmetry curve, 1 = Cubic curve, 2 = User-defined curve */
typedef enum {
    HI_ISP_DRC_CURVE_ASYMMETRY = 0x0,
    HI_ISP_DRC_CURVE_USER,
    HI_ISP_DRC_CURVE_BUTT
} hi_isp_drc_curve_select;

typedef struct {
    hi_bool enable;
    hi_isp_drc_curve_select curve_select;  /* RW; Range:[0x0, 0x2]; Select tone mapping curve type */
    hi_u8  purple_reduction_strength; /* RW; Range:[0x0, 0x80]; Format:8.0; Controls the purple detection strength */
    hi_u8  local_mixing_bright_max; /* RW; Range:[0x0, 0x80]; Format:8.0;
                                       Maximum enhancement strength for positive detail */
    hi_u8  local_mixing_bright_min; /* RW; Range:[0x0, 0x40]; Format:8.0;
                                       Minimum enhancement strength for positive detail */
    hi_u8  local_mixing_bright_thr; /* RW; Range:[0x0, 0xFF]; Format:8.0;
                                       Luma threshold for enhancement strength adaptation of positive detail */
    hi_s8  local_mixing_bright_slo; /* RW; Range:[-7, 7]; Format:s4.0;
                                       Enhancement strength adaptation slope for positive detail */
    hi_u8  local_mixing_dark_max;   /* RW; Range:[0x0, 0x80]; Format:8.0;
                                       Maximum enhancement strength for negative detail */
    hi_u8  local_mixing_dark_min;   /* RW; Range:[0x0, 0x40]; Format:8.0;
                                       Minimum enhancement strength for negative detail */
    hi_u8  local_mixing_dark_thr;   /* RW; Range:[0x0, 0xFF]; Format:8.0;
                                       Luma threshold for enhancement strength adaptation of negative detail */
    hi_s8  local_mixing_dark_slo;   /* RW; Range:[-7, 7]; Format:s4.0;
                                       Enhancement strength adaptation slope for negative detail */

    hi_u8  bright_gain_lmt;         /* RW; Range:[0x0, 0xF];  Format:4.0; Bright area gain high limit */
    hi_u8  bright_gain_lmt_step;    /* RW; Range:[0x0, 0xF];  Format:4.0; Bright area gain high limit step */
    hi_u8  dark_gain_lmt_y;         /* RW; Range:[0x0, 0x85]; Format:7.0; Dark area luma gain limit */
    hi_u8  dark_gain_lmt_c;         /* RW; Range:[0x0, 0x85]; Format:7.0; Dark area chroma gain limit */
    hi_u16 color_correction_lut[HI_ISP_DRC_CC_NODE_NUM]; /* RW; Range:[0x0, 0x400]; Format:4.12;
                                                            LUT of color correction coefficients */
    hi_u16 tone_mapping_value[HI_ISP_DRC_TM_NODE_NUM];   /* RW; Range:[0x0, 0xffff]; Format:16.0;
                                                            LUT of user-defined curve */

    hi_u8  contrast_control;     /* RW; Range:[0x0, 0xF]; Format:4.0; Contrast control */
    hi_s8  detail_adjust_factor; /* RW; Range:[-15, 15];  Format:4.0; Detail adjustment factor */

    hi_u8  spatial_flt_coef;     /* RW; Range: [0x0, 0x5] Spatial filter coefficients */
    hi_u8  range_flt_coef;    /* RW; Range:[0x0, 0xA]; Format:4.0; Range filter coefficients */
    hi_u8  range_ada_max;     /* RW; Range:[0x0, 0x8]; Format:4.0; Maximum range filter coefficient adaptation range */

    hi_u8  grad_rev_max;         /* RW; Range:[0x0, 0x40]; Format:7.0; Maximum gradient reversal reduction strength */
    hi_u8  grad_rev_thr;         /* RW; Range:[0x0, 0x80]; Format:8.0; Gradient reversal reduction threshold */

    hi_op_mode op_type;
    hi_isp_drc_manual_attr manual_attr;
    hi_isp_drc_auto_attr   auto_attr;
    hi_isp_drc_asymmetry_curve_attr asymmetry_curve;
} hi_isp_drc_attr;

typedef struct {
    hi_u8 wgt;   /* RW;Range: [0x0, 0xFF];Format:1.7;Weight of Gaussian distribution */
    hi_u8 sigma; /* RW;Range: [0x1, 0xFF];Format:0.8;Sigma of Gaussian distribution */
    hi_u8 mean;  /* RW;Range: [0x0, 0xFF];Format:0.8;Mean of Gaussian distribution */
} hi_isp_ldci_gauss_coef_attr;

typedef struct {
    hi_isp_ldci_gauss_coef_attr  he_pos_wgt;
    hi_isp_ldci_gauss_coef_attr  he_neg_wgt;
} hi_isp_ldci_he_wgt_attr;

typedef struct {
    hi_isp_ldci_he_wgt_attr  he_wgt;
    hi_u16                   blc_ctrl;  /* RW;Range: [0x0, 0x1FF];Format:9.0;Restrain dark region */
} hi_isp_ldci_manual_attr;

typedef struct {
    hi_isp_ldci_he_wgt_attr  he_wgt[HI_ISP_AUTO_ISO_NUM];
    hi_u16                   blc_ctrl[HI_ISP_AUTO_ISO_NUM]; /* RW;Range: [0x0, 0x1FF];Format:9.0;auto mode,
                                                                     Restrain dark region */
} hi_isp_ldci_auto_attr;

typedef struct {
    hi_bool                 enable;          /* RW;Range:[0x0,0x1];Format:1.0; */
    hi_u8                   gauss_lpf_sigma; /* RW;Range: [0x1, 0xFF];Format:0.8;
                                                Coefficient of Gaussian low-pass filter */
    hi_op_mode          op_type;
    hi_isp_ldci_manual_attr manual_attr;
    hi_isp_ldci_auto_attr   auto_attr;
    hi_u16                  tpr_incr_coef;   /* RW;Range: [0x0, 0x100];Format:0.9;
                                                Increase Coefficient of temporal filter */
    hi_u16                  tpr_decr_coef;   /* RW;Range: [0x0, 0x100];Format:0.9;
                                                Decrease Coefficient of temporal filter */
} hi_isp_ldci_attr;

typedef struct {
    hi_u32  y_ratio_lut[HI_ISP_CA_YRATIO_LUT_LENGTH];  /* RW;Range:[0,2047];Format:1.11 */
    hi_s32  iso_ratio[HI_ISP_AUTO_ISO_NUM];      /* RW;Range:[0,2047];Format:1.10 */
} hi_isp_ca_lut;

typedef struct {
    hi_bool enable;             /* RW;Range:[0x0,0x1];Format:1.0 */
    hi_isp_ca_lut  ca;
} hi_isp_ca_attr;

/* CSC */
typedef struct {
    hi_s16 csc_in_dc[HI_ISP_CSC_DC_NUM]; /* RW; Range:[-1024, 1023]; Format: 11.0; Input dc component for csc matrix */
    hi_s16 csc_out_dc[HI_ISP_CSC_DC_NUM]; /* RW; Range:[-1024, 1023]; Format: 11.0;Output dc component for csc matrix */
    hi_s16 csc_coef[HI_ISP_CSC_COEF_NUM]; /* RW; Range:[-4096, 4095]; Format: 5.10; 3x3 coefficients for csc matrix */
} hi_isp_csc_matrix;

typedef struct {
    hi_bool enable;            /* RW; Range:[0, 1];Format:1.0; Enable/Disable CSC Function */

    hi_color_gamut color_gamut; /* RW; Range: [0, 3]; Color gamut type; COLOR_GAMUT_BT2020 not support */
    hi_u8   hue;              /* RW; Range:[0, 100];Format:8.0; Csc Hue Value */
    hi_u8   luma;             /* RW; Range:[0, 100];Format:8.0; Csc Luma Value */
    hi_u8   contr;            /* RW; Range:[0, 100];Format:8.0; Csc Contrast Value */
    hi_u8   satu;             /* RW; Range:[0, 100];Format:8.0; Csc Saturation Value */
    hi_bool limited_range_en; /* RW; Range: [0x0, 0x1];
                                 Enable/Disable: Enable Limited range output mode(default full range output) */
    hi_bool ext_csc_en;       /* RW; Range: [0x0, 0x1]; Enable/Disable: Enable extended luma range */
    hi_bool ct_mode_en;       /* RW; Range: [0x0, 0x1]; Enable/Disable: Enable ct mode */
    hi_isp_csc_matrix csc_magtrx;     /* RW; Color Space Conversion matrix */
} hi_isp_csc_attr;

/* CLUT */
typedef struct {
    hi_bool enable; /* RW; Range:[0, 1];Format:1.0; Enable/Disable CLUT Function */
    hi_u32  gain_r; /* RW; Range:[0, 4095];Format:12.0 */
    hi_u32  gain_g; /* RW; Range:[0, 4095];Format:12.0 */
    hi_u32  gain_b; /* RW; Range:[0, 4095];Format:12.0 */
} hi_isp_clut_attr;

/* CLUT LUT */
typedef struct {
    hi_u32 lut[HI_ISP_CLUT_LUT_LENGTH];  /* RW; Range:[0x0, 1073741823]; */
} hi_isp_clut_lut;

/*
  Defines the type of static defect pixel calibration
  0 = bright defect pixel calibration
  1 = dark defect pixel calibration
 */
typedef enum {
    HI_ISP_STATIC_DP_BRIGHT = 0x0,
    HI_ISP_STATIC_DP_DARK,
    HI_ISP_STATIC_DP_BUTT
} hi_isp_static_dp_type;

typedef struct {
    hi_bool enable_detect;                /* RW; Range: [0, 1];Format 1.0;Set 'HI_TRUE'to start static defect-pixel
                                             calibration, and firmware will set 'HI_FALSE' when finished */
    hi_isp_static_dp_type static_dp_type; /* RW; Range: [0, 1];Format 1.0;Select static bright/dark defect-pixel
                                             calibration */
    hi_u8  start_thresh;  /* RW; Range: [1, 255]; Format 8.0;Start threshold for static defect-pixel calibraiton */
    hi_u16 count_max;     /* RW; Range:[0, 6144];Format 14.0; limited Range:
                             [0, HI_ISP_STATIC_DP_COUNT_NORMAL*BlkNum],
                             Limit of max number of static defect-pixel calibraiton. */
    hi_u16 count_min;     /* RW; Range: [0, 6144];Format 14.0; limited Range: [0, u16CountMax],
                             Limit of min number of static defect-pixel calibraiton. */
    hi_u16 time_limit;    /* RW; Range: [0x0, 1600];Format 11.0;Time limit for static defect-pixel calibraiton, in
                             frame number. */

    hi_u32 table[HI_ISP_STATIC_DP_COUNT_MAX]; /* R; Range: [0, 0x1FFF1FFF];Format 29.0;Static
                                              defect-pixel calibraiton table,
                                              0~12 bits represents the X coordinate of the defect pixel,
                                              16~28 bits represent the Y coordinate of the defect pixel. */
    hi_u8  finish_thresh;   /* R; Range: [0, 255];Format 8.0; Finish threshold for static defect-pixel calibraiton. */
    hi_u16 count;          /* R; Range:[0, 6144]; Format 14.0; Finish number for static defect-pixel calibraiton. */
    hi_isp_status status;  /* R; Range: [0, 2];Format 2.0;Status of static defect-pixel calibraiton. */
} hi_isp_dp_static_calibrate;

typedef struct {
    hi_bool enable;       /* RW; Range:[0, 1]; Format 1.0; Enable/disable the static defect-pixel module. */
    hi_u16  bright_count; /* RW; Range:[0, 6144]; Format 14.0; limited Range:[0, HI_ISP_STATIC_DP_COUNT_NORMAL*BlkNum],
                             When used as input(W), indicate the number of static bright defect pixels;
                             As output(R),indicate the number of static bright and dark defect pixels. */
    hi_u16  dark_count;   /* RW; Range:[0, 6144]; Format 14.0; limited Range:[0, HI_ISP_STATIC_DP_COUNT_NORMAL*BlkNum],
                             When used as input(W), indicate the number of static dark defect pixels;
                             As output(R), invalid value 0. */
    hi_u32  bright_table[HI_ISP_STATIC_DP_COUNT_MAX]; /* RW; Range: [0x0, 0x1FFF1FFF]; Format 29.0;
                                                  0~12 bits represents the X coordinate of the defect pixel,
                                                  16~28 bits represent the Y coordinate of the defect pixel
                                                  Notice: When used as input(W), indicate static bright defect pixels
                                                  table; As output(R), indicate static bright and dark defect pixels
                                                  table. */

    hi_u32  dark_table[HI_ISP_STATIC_DP_COUNT_MAX]; /* RW; Range: [0x0, 0x1FFF1FFF]; Format 29.0;
                                                0~12 bits represents the X coordinate of the defect pixel,
                                                16~28 bits represent the Y coordinate of the defect pixel
                                                Notice: When used as input(W), indicate static dark defect pixels table;
                                                As output(R), invalid value. */
    hi_bool show;                  /* RW; Range: [0, 1]; Format 1.0; RW;highlight static defect pixel. */
} hi_isp_dp_static_attr;

typedef struct {
    hi_u16  strength;           /* RW; Range: [0, 255];Format:8.0;Dynamic DPC strength. */
    hi_u16  blend_ratio;        /* RW; Range: [0, 128];Format:9.0;Blending ratio required for DPC */
} hi_isp_dp_dynamic_manual_attr;

typedef struct {
    hi_u16  strength[HI_ISP_AUTO_ISO_NUM];  /* RW; Range: [0, 255];Format:8.0;Dynamic DPC strength. */
    hi_u16  blend_ratio[HI_ISP_AUTO_ISO_NUM]; /* RW; Range: [0, 128];Format:9.0;Blending ratio required for DPC */
} hi_isp_dp_dynamic_auto_attr;

typedef struct {
    hi_bool enable;             /* RW; Range: [0, 1];Format 1.0;Enable/disable the dynamic defect-pixel module */
    hi_bool sup_twinkle_en;     /* RW; Range: [0, 1];Format 1.0;Enable/disable the twinkle suppression  module */
    hi_s8   soft_thr;           /* RW; Range: [0, 127];Format s8.0;twinkle suppression threshold */
    hi_u8   soft_slope;         /* RW; Range: [0, 255];Format 8.0;Correction controlling parameter of the pixels whose
                                   deviation is smaller than s8SoftThr. */
    hi_op_mode op_type;     /* RW; Range: [0, 1];Format 1.0;Working mode of dynamic DPC */
    hi_isp_dp_dynamic_manual_attr manual_attr;
    hi_isp_dp_dynamic_auto_attr   auto_attr;
} hi_isp_dp_dynamic_attr;

typedef struct {
    hi_bool enable; /* RW; Range:[0, 1];Format 1.0;Enable/disable dis module. Not support. Please check hi_dis_attr. */
} hi_isp_dis_attr;

typedef struct {
    hi_bool enable;      /* RW; Range:[0, 1];Format:1.0; HI_TRUE: enable lsc; HI_FALSE: disable lsc */
    hi_u16  mesh_strength;    /* RW; Range:[0, 65535]; The strength of the mesh shading correction */
    hi_u16  blend_ratio; /* RW; Range:[0, 256];Format:9.0; the blendratio of the two mesh gain lookup-table */
} hi_isp_shading_attr;

typedef struct {
    hi_u16  r_gain[HI_ISP_LSC_GRID_POINTS];  /* RW; Range:[0, 1023];Member used to store the calibration data of the
                                                R channel required for LSC. */
    hi_u16  gr_gain[HI_ISP_LSC_GRID_POINTS]; /* RW; Range:[0, 1023];Member used to store the calibration data of the
                                                Gr channel required for LSC. */
    hi_u16  gb_gain[HI_ISP_LSC_GRID_POINTS]; /* RW; Range:[0, 1023];Member used to store the calibration data of the
                                                Gb channel required for LSC. */
    hi_u16  b_gain[HI_ISP_LSC_GRID_POINTS];  /* RW; Range:[0, 1023];Member used to store the calibration data of the
                                                B channel required for LSC. */
} hi_isp_shading_gain_lut;

typedef struct {
    hi_u16  r_gain[HI_ISP_RLSC_POINTS];  /* RW; Range:[0, 65535];Member used to store the calibration data of the
                                            R channel required for BNR_LSC */
    hi_u16  gr_gain[HI_ISP_RLSC_POINTS]; /* RW; Range:[0, 65535];Member used to store the calibration data of the
                                            Gr channel required for BNR_LSC */
    hi_u16  gb_gain[HI_ISP_RLSC_POINTS]; /* RW; Range:[0, 65535];Member used to store the calibration data of the
                                            Gb channel required for BNR_LSC */
    hi_u16  b_gain[HI_ISP_RLSC_POINTS];  /* RW; Range:[0, 65535];Member used to store the calibration data of the
                                            B channel required for BNR_LSC */
} hi_isp_bnr_lsc_gain_lut;

typedef struct {
    hi_u8   mesh_scale;        /* RW; Range:[0, 7];Format:3.0;Three bit value that selects the scale and precision for
                                  the 10 bit gain coefficients stored in mesh tables */
    hi_u16  x_grid_width[HI_ISP_MLSC_X_HALF_GRID_NUM]; /* RW; Range:[4, 1092]; limited Range:[4, u32Width/4 - 60],
                                                          Member used to store the width data of each GRID partition */
    hi_u16  y_grid_width[HI_ISP_MLSC_Y_HALF_GRID_NUM]; /* RW; Range:[4, 1092]; limited Range:[4, u32Height/4 - 60];
                                                        Member used to store the height data of each GRID partition */
    hi_isp_shading_gain_lut lsc_gain_lut[HI_ISP_MLSC_GAIN_LUT_NUM];
    hi_isp_bnr_lsc_gain_lut bnr_lsc_gain_lut;
} hi_isp_shading_lut_attr;

typedef struct {
    hi_isp_bayer_format bayer_format;    /* RW; Range: [0, 3];Format ENUM;Shows bayer pattern */
    hi_isp_bayer_raw_bit raw_bit;  /* RW; Range: {8,10,12,14,16};Format ENUM;Shows input raw bitwidth */

    hi_u16  img_height; /* RW; Range: [0, 65535];Format 16.0;Input raw image height */
    hi_u16  img_width;  /* RW; Range: [0, 65535];Format 16.0;Input raw image width */

    hi_u16  dst_img_height; /* RW; Range: [0, 65535];Format 16.0;limited Range:[0, u16ImgHeight], Image height that
                               crop from input raw image, set to ImgHeight if don't need to crop */
    hi_u16  dst_img_width;  /* RW; Range: [0, 65535];Format 16.0;limited Range:[0, u16ImgWidth], Image width that
                               crop from input raw image, set to ImgWidth if don't need to crop */
    hi_u16  offset_x;       /* RW; Range: [0, 65535];Format 16.0;limited Range:[0, u16ImgWidth - u16DstImgWidth],
                               Horizontal offset that crop from input raw image, set to 0 if don't need to crop */
    hi_u16  offset_y;       /* RW; Range: [0, 65535];Format 16.0;limited Range:[0, u16ImgHeight - u16DstImgHeight],
                               Vertical offset that crop from input raw image, set to 0 if don't need to crop */

    hi_u32  mesh_scale; /* RW; Range: [0, 7];Format 3.0; Shows Mesh Scale value */

    hi_u16  blc_offset_r; /* RW; Range: [0, 4095];Format 12.0;BLC value for R channel */
    hi_u16  blc_offset_gr; /* RW; Range: [0, 4095];Format 12.0;BLC value for Gr channel */
    hi_u16  blc_offset_gb; /* RW; Range: [0, 4095];Format 12.0;BLC value for Gb channel */
    hi_u16  blc_offset_b; /* RW; Range: [0, 4095];Format 12.0;BLC value for B channel */
} hi_isp_mlsc_calibration_cfg;

typedef struct {
    hi_u8   mesh_scale;       /* RW; Range:[0, 7];Format:3.0;Three bit value that selects the scale and precision for
                                 the 10 bit gain coefficients stored in mesh tables */
    hi_u16  x_grid_width[HI_ISP_MLSC_X_HALF_GRID_NUM]; /* RW; Range:[4, 16323]; limited Range:[4, u16ImgWidth /4 - 60],
                                                     Member used to store the width data of each GRID partition */
    hi_u16  y_grid_width[HI_ISP_MLSC_Y_HALF_GRID_NUM]; /* RW; Range:[4, 16323]; limited Range:[4, u16ImgHeight /4 - 60],
                                                     Member used to store the height data of each GRID partition */
    hi_isp_shading_gain_lut lsc_gain_lut;
    hi_isp_bnr_lsc_gain_lut bnr_lsc_gain_lut;
} hi_isp_mesh_shading_table;

/* Not Support */
typedef struct {
    hi_bool enable;          /* RW; Range:[0, 1];Format:1.0; enable/disable rlsc */
    hi_u16  radial_str;     /* RW; Range:[0, 65535];Format:4.12; the strength of the mesh shading correction */
} hi_isp_radial_shading_attr;

/* Not Support */
typedef struct {
    hi_u16  r_gain[HI_ISP_RLSC_POINTS];  /* RW; Range:[0, 65535];Member used to store the calibration data of the
                                            R channel required for RLSC */
    hi_u16  gr_gain[HI_ISP_RLSC_POINTS]; /* RW; Range:[0, 65535];Member used to store the calibration data of the
                                            Gr channel required for RLSC */
    hi_u16  gb_gain[HI_ISP_RLSC_POINTS]; /* RW; Range:[0, 65535];Member used to store the calibration data of the
                                            Gb channel required for RLSC */
    hi_u16  b_gain[HI_ISP_RLSC_POINTS];  /* RW; Range:[0, 65535];Member used to store the calibration data of the
                                            B channel required for RLSC */
} hi_isp_radial_shading_gain_lut;

/* Not Support */
typedef struct {
    hi_op_mode light_mode;  /* RW; Range:[0, 1];Format:1.0; 1: manual mode; 0: auto mode */
    hi_u16  blend_ratio;           /* RW; Range:[0, 256];Format:8.0; Used in manual mode only, indicates the light
                                      blending strength for the first light info */
    hi_u8   light_type1;           /* RW; Range:[0, 2];Format:2.0; Used in manual mode only, indicates the first light
                                      source selected */
    hi_u8   light_type2;           /* RW; Range:[0, 2];Format:2.0; Used in manual mode only, indicates the second light
                                      source selected */
    hi_u8   radial_scale;          /* RW; Range:[0, 13];Format:4.0;Four bit value that selects the scale and precision
                                      for the 10 bit gain coefficients */
    hi_u16  center_r_x;    /* RW; Range:[0, 65535];Format:16.0;Limited Range:[0, Width], shows x value of the position
                              of optical center of red channel */
    hi_u16  center_r_y;    /* RW; Range:[0, 65535];Format:16.0;Limited Range:[0, Height], shows y value of the position
                              of optical center of red channel */
    hi_u16  center_gr_x;   /* RW; Range:[0, 65535];Format:16.0;Limited Range:[0, Width], shows x value of the position
                              of optical center of gr channel */
    hi_u16  center_gr_y;   /* RW; Range:[0, 65535];Format:16.0;Limited Range:[0, Height], shows y value of the position
                              of optical center of gr channel */
    hi_u16  center_gb_x;   /* RW; Range:[0, 65535];Format:16.0;Limited Range:[0, Width], shows x value of the position
                              of optical center of gb channel */
    hi_u16  center_gb_y;   /* RW; Range:[0, 65535];Format:16.0;Limited Range:[0, Height], shows y value of the position
                              of optical center of gb channel */
    hi_u16  center_b_x;    /* RW; Range:[0, 65535];Format:16.0;Limited Range:[0, Width], shows x value of the position
                              of optical center of blue channel */
    hi_u16  center_b_y;    /* RW; Range:[0, 65535];Format:16.0;Limited Range:[0, Height], shows y value of the position
                              of optical center of blue channel */
    hi_u16  off_center_r;  /* RW; Range:[0, 65535];Format:16.0;related to the 1/R^2 value of red channel */
    hi_u16  off_center_gr; /* RW; Range:[0, 65535];Format:16.0;related to the 1/R^2 value of gr channel */
    hi_u16  off_center_gb; /* RW; Range:[0, 65535];Format:16.0;related to the 1/R^2 value of gb channel */
    hi_u16  off_center_b;  /* RW; Range:[0, 65535];Format:16.0;related to the 1/R^2 value of blue channel */
    hi_isp_radial_shading_gain_lut rlsc_gain_lut[3]; /* Not Support */
} hi_isp_radial_shading_lut_attr;

typedef struct {
    hi_isp_bayer_format bayer;    /* RW; Range: [0, 3];Format ENUM;Shows bayer pattern */
    hi_isp_bayer_raw_bit raw_bit;  /* RW; Range: {8,10,12,14,16};Format ENUM;Shows input raw bitwidth */

    hi_u16  img_height; /* RW; Range: [0, 65535];Format 16.0;Input raw image height */
    hi_u16  img_width;  /* RW; Range: [0, 65535];Format 16.0;Input raw image width */

    hi_u16  center_x;       /* RW; Range: [0, 65535];Format 16.0;limited Range:[0, u16ImgWidth - u16DstImgWidth],
                               Horizontal offset that crop from input raw image, set to 0 if don't need to crop */
    hi_u16  center_y;       /* RW; Range: [0, 65535];Format 16.0;limited Range:[0, u16ImgHeight - u16DstImgHeight],
                               Vertical offset that crop from input raw image, set to 0 if don't need to crop */
    hi_u16  radius;      /* RW; Range: [0, 65535];Format 16.0; radius of the area need to calibrate */

    hi_u16  radial_scale; /* RW; Range: [0, 13];Format 3.0; Shows Scale value */

    hi_u16  blc_offset_r; /* RW; Range: [0, 4095];Format 12.0;BLC value for R channel */
    hi_u16  blc_offset_gr; /* RW; Range: [0, 4095];Format 12.0;BLC value for Gr channel */
    hi_u16  blc_offset_gb; /* RW; Range: [0, 4095];Format 12.0;BLC value for Gb channel */
    hi_u16  blc_offset_b; /* RW; Range: [0, 4095];Format 12.0;BLC value for B channel */
} hi_isp_rlsc_calibration_cfg;

typedef struct {
    hi_u8   radial_scale;       /* RW; Range:[0, 13];Format:4.0;Four bit value that selects the scale and precision
                                      for the 10 bit gain coefficients */
    hi_u16  center_x;       /* RW; Range:[0, 65535];Format:16.0;Limited Range:[0, Width], shows x value of the position
                              of optical center */
    hi_u16  center_y;    /* RW; Range:[0, 65535];Format:16.0;Limited Range:[0, Height], shows y value of the position
                              of optical center */
    hi_u16  offcenter;   /* RW; Range:[0, 65535];Format:16.0;related to the 1/R^2 value of all channel */
    hi_isp_radial_shading_gain_lut rlsc_gain_lut; /* Not Support */
} hi_isp_radial_shading_table;

typedef struct {
    hi_bool enable;       /* RW; Range:[0,1];Format:1.0;Acs Enable */
    hi_u16  y_strength;   /* RW; Range:[0,256];Format:8.0;Acs correction strength for y shading */
    hi_u16  run_interval; /* RW; Range:[1,255];Format:16.0;Acs Run Interval */
    hi_bool lock_enable;  /* RW; Range:[0,1];Format:1.0;Lock Gain Lut Enable */
} hi_isp_acs_attr;

typedef struct {
    hi_u8   chroma_strength[HI_ISP_BAYER_CHN_NUM]; /* RW;Range:[0, 3];Format:2.0;
                                                      Strength of Chrmoa noise reduction for R/Gr/Gb/B channel */
    hi_u8   fine_strength;                      /* RW;Range:[0, 0x80];Format:8.0;Strength of Luma noise reduction */
    hi_u16  coring_wgt;                     /* RW;Range:[0, 0xc80];Format:12.0;Strength of reserving the random noise */
    hi_u16  coarse_strength[HI_ISP_BAYER_CHN_NUM]; /* RW;Range:[0, 0x360];Format:10.0;
                                                   Coarse Strength of noise reduction */
} hi_isp_nr_manual_attr;

typedef struct {
    hi_u8   chroma_strength[HI_ISP_BAYER_CHN_NUM][HI_ISP_AUTO_ISO_NUM];  /* RW;Range:[0, 3];Format:2.0;
                                                                          Strength of chrmoa noise reduction for
                                                                          R/Gr/Gb/B channel */
    hi_u8   fine_strength[HI_ISP_AUTO_ISO_NUM];                      /* RW;Range:[0, 0x80];Format:8.0;
                                                                         Strength of luma noise reduction */
    hi_u16  coring_wgt[HI_ISP_AUTO_ISO_NUM];                    /* RW;Range:[0, 0xc80];Format:12.0;
                                                                         Strength of reserving the random noise */
    hi_u16  coarse_strength[HI_ISP_BAYER_CHN_NUM][HI_ISP_AUTO_ISO_NUM]; /* RW;Range:[0, 0x360];Format:10.0;
                                                                         Coarse Strength of noise reduction */
} hi_isp_nr_auto_attr;

typedef struct {
    hi_u8 wdr_frame_strength[HI_ISP_WDR_MAX_FRAME_NUM];    /* RW;Range:[0, 0x50];Format:7.0;
                                                          Strength of each frame in wdr mode */
    hi_u8 fusion_frame_strength[HI_ISP_WDR_MAX_FRAME_NUM]; /* RW;Range:[0, 0x50];Format:7.0;
                                                          Strength of each frame in wdr mode */
} hi_isp_nr_wdr_attr;

typedef struct {
    hi_bool  enable;               /* RW;Range:[0, 1];Format:1.0; Nr Enable */
    hi_bool  nr_lsc_enable;        /* RW;Range:[0, 1];Format:1.0;
                                      HI_TRUE: Noise reduction refers to lens shading;
                                      HI_FALSE: Noise reduction not refers to lens shading; */
    hi_u8    bnr_lsc_max_gain;     /* RW;Range:[0, 0xbf];Format:2.6; Max gain for referring to lens shading */
    hi_u16   bnr_lsc_cmp_strength; /* RW;Range:[0, 0x100];Format:1.8; Compare strength for referring to lens shading */
    hi_u16   coring_ratio[HI_ISP_BAYERNR_LUT_LENGTH];  /* RW;Range:[0, 0x3ff];Format:12.0;
                                                          Strength of reserving the random noise according to luma */

    hi_op_mode op_type;
    hi_isp_nr_auto_attr auto_attr;
    hi_isp_nr_manual_attr manual_attr;
    hi_isp_nr_wdr_attr  wdr_attr;
} hi_isp_nr_attr;

typedef struct {
    hi_u16       global_gain;         /* RW;Range:[0, 0x100];Format:1.8 */
    hi_u16       gain_lf;             /* RW;Range:[0, 0x20];Format:2.4 */
    hi_u16       gain_hf;             /* RW;Range:[0, 0x20];Format:2.4 */
} hi_isp_de_manual_attr;

typedef struct {
    hi_u16       global_gain[HI_ISP_AUTO_ISO_NUM]; /* RW;Range:[0x0, 0x100];Format:1.8 */
    hi_u16       gain_lf[HI_ISP_AUTO_ISO_NUM];     /* RW;Range:[0x0, 0x20];Format:2.4 */
    hi_u16       gain_hf[HI_ISP_AUTO_ISO_NUM];     /* RW;Range:[0x0, 0x20];Format:2.4 */
} hi_isp_de_auto_attr;

typedef struct {
    hi_bool       enable;                                    /* RW;Range:[0x0,0x1];Format:1.0; De Enable */
    hi_u16        luma_gain_lut[HI_ISP_DE_LUMA_GAIN_LUT_N]; /* RW;Range:[0x0,0x100];Format:1.8 */

    hi_op_mode        op_type;
    hi_isp_de_auto_attr   auto_attr;
    hi_isp_de_manual_attr manual_attr;
} hi_isp_de_attr;

/*
  Defines the type of the ISP gamma curve
  0 = Default curve
  1 = Default BT.709 curve (Gamma Curve)
  2 = Default SMPTE.2084 curve (PQ Curve), not supported
  3 = User defined Gamma curve, LUT must be correct
*/
typedef enum {
    HI_ISP_GAMMA_CURVE_DEFAULT = 0x0,
    HI_ISP_GAMMA_CURVE_SRGB,
    HI_ISP_GAMMA_CURVE_HDR,    /* Not supported */
    HI_ISP_GAMMA_CURVE_USER_DEFINE,
    HI_ISP_GAMMA_CURVE_BUTT
} hi_isp_gamma_curve_type;

typedef struct {
    hi_bool   enable;                     /* RW; Range:[0, 1]; Format:1.0;Enable/Disable Gamma Function */
    hi_u16    table[HI_ISP_GAMMA_NODE_NUM];    /* RW; Range:[0, 4095]; Format:12.0;Gamma LUT nodes value */
    hi_isp_gamma_curve_type curve_type;    /* RW; Range:[0, 3]; Format:2.0;Gamma curve type */
} hi_isp_gamma_attr;

typedef struct {
    hi_bool enable;                   /* RW; Range:[0, 1]; Format:1.0;Enable/Disable PreGamma Function */
    hi_u32  table[HI_ISP_PREGAMMA_NODE_NUM]; /* RW; Range: [0, 0xFFFFF]; Format:21.0; PreGamma LUT nodes value */
} hi_isp_pregamma_attr;

typedef struct {
    hi_u8  luma_wgt[HI_ISP_SHARPEN_LUMA_NUM]; /* RW; Range:[0, 127];
                                               Format:0.7;Adjust the sharpen strength according to luma.
                                               Sharpen strength will be weaker when it decrease. */
    hi_u16 texture_strength[HI_ISP_SHARPEN_GAIN_NUM]; /* RW; Range: [0, 4095]; Format:7.5;Undirectional sharpen
                                                         strength for texture and detail enhancement */
    hi_u16 edge_strength[HI_ISP_SHARPEN_GAIN_NUM];    /* RW; Range: [0, 4095]; Format:7.5;
                                           Directional sharpen strength for edge enhancement */
    hi_u16 texture_freq;       /* RW; Range: [0, 4095];Format:6.6; Texture frequency adjustment.
                                  Texture and detail will be finer when it increase */
    hi_u16 edge_freq;          /* RW; Range: [0, 4095];Format:6.6; Edge frequency adjustment.
                                  Edge will be narrower and thiner when it increase */
    hi_u8  over_shoot;         /* RW; Range: [0, 127]; Format:7.0;u8OvershootAmt */
    hi_u8  under_shoot;        /* RW; Range: [0, 127]; Format:7.0;u8UndershootAmt */
    hi_u8  shoot_sup_strength;  /* RW; Range: [0, 255]; Format:8.0;overshoot and undershoot suppression strength, the
                                  amplitude and width of shoot will be decrease when shootSupSt increase */
    hi_u8  shoot_sup_adj;      /* RW; Range: [0, 15]; Format:4.0;overshoot and undershoot suppression adjusting,
                                  adjust the edge shoot suppression strength */
    hi_u8  detail_ctrl;        /* RW; Range: [0, 255]; Format:8.0;Different sharpen strength for detail and edge.
                                  When it is bigger than 128, detail sharpen strength will be stronger than edge */
    hi_u8  detail_ctrl_threshold;    /* RW; Range: [0, 255]; Format:8.0; The threshold of DetailCtrl, it is used to
                                  distinguish detail and edge. */
    hi_u8  edge_filt_strength;      /* RW; Range: [0, 63]; Format:6.0;The strength of edge filtering. */
    hi_u8  edge_filt_max_cap;  /* RW; Range: [0, 47]; Format:6.0;The max capacity of edge filtering. */
    hi_u8  r_gain;             /* RW; Range: [0, 31];   Format:5.0;Sharpen Gain for Red Area */
    hi_u8  g_gain;             /* RW; Range: [0, 255]; Format:8.0; Sharpen Gain for Green Area */
    hi_u8  b_gain;             /* RW; Range: [0, 31];   Format:5.0;Sharpen Gain for Blue Area */
    hi_u8  skin_gain;          /* RW; Range: [0, 31]; Format:5.0;Sharpen Gain for Skin Area */
    hi_u16 max_sharp_gain;     /* RW; Range: [0, 0x7FF]; Format:8.3; Maximum sharpen gain */
} hi_isp_sharpen_manual_attr;

typedef struct {
    hi_u8  luma_wgt[HI_ISP_SHARPEN_LUMA_NUM][HI_ISP_AUTO_ISO_NUM];  /* RW; Range:[0, 127].
                                                                    Sharpen strength will be weaker when it decrease */
    hi_u16 texture_strength[HI_ISP_SHARPEN_GAIN_NUM][HI_ISP_AUTO_ISO_NUM];  /* RW; Range: [0, 4095]; Format:7.5;
                                                                             Undirectional sharpen strength for texture
                                                                             and detail enhancement */
    hi_u16 edge_strength[HI_ISP_SHARPEN_GAIN_NUM][HI_ISP_AUTO_ISO_NUM]; /* RW; Range: [0, 4095]; Format:7.5; Directional
                                                                         sharpen strength for edge enhancement */
    hi_u16 texture_freq[HI_ISP_AUTO_ISO_NUM];   /* RW; Range: [0, 4095]; Format:6.6;Texture frequency adjustment.
                                                         Texture and detail will be finer when it increase */
    hi_u16 edge_freq[HI_ISP_AUTO_ISO_NUM];      /* RW; Range: [0, 4095]; Format:6.6;Edge frequency adjustment.
                                                         Edge will be narrower and thiner when it increase */
    hi_u8  over_shoot[HI_ISP_AUTO_ISO_NUM];     /* RW; Range: [0, 127];  Format:7.0;u8OvershootAmt */
    hi_u8  under_shoot[HI_ISP_AUTO_ISO_NUM];    /* RW; Range: [0, 127];  Format:7.0;u8UndershootAmt */
    hi_u8  shoot_sup_strength[HI_ISP_AUTO_ISO_NUM];  /* RW; Range: [0, 255]; Format:8.0;overshoot and undershoot
                                                         suppression strength, the amplitude and width of shoot will be
                                                         decrease when shootSupSt increase */
    hi_u8  shoot_sup_adj[HI_ISP_AUTO_ISO_NUM];  /* RW; Range: [0, 15]; Format:4.0;overshoot and undershoot
                                                   suppression adjusting, adjust the edge shoot suppression strength */
    hi_u8  detail_ctrl[HI_ISP_AUTO_ISO_NUM];    /* RW; Range: [0, 255]; Format:8.0;Different sharpen strength for
                                                         detail and edge. When it is bigger than 128, detail sharpen
                                                         strength will be stronger than edge. */
    hi_u8  detail_ctrl_threshold[HI_ISP_AUTO_ISO_NUM]; /* RW; Range: [0, 255]; Format:8.0; The threshold of DetailCtrl,
                                                          it is used to distinguish detail and edge. */
    hi_u8  edge_filt_strength[HI_ISP_AUTO_ISO_NUM];   /* RW; Range: [0, 63]; Format:6.0;
                                                          The strength of edge filtering. */
    hi_u8  edge_filt_max_cap[HI_ISP_AUTO_ISO_NUM]; /* RW; Range: [0, 47]; Format:6.0;
                                                            The max capacity of edge filtering. */
    hi_u8  r_gain[HI_ISP_AUTO_ISO_NUM];        /* RW; Range: [0, 31]; Format:5.0; Sharpen Gain for Red Area */
    hi_u8  g_gain[HI_ISP_AUTO_ISO_NUM];        /* RW; Range: [0, 255]; Format:8.0; Sharpen Gain for Green Area */
    hi_u8  b_gain[HI_ISP_AUTO_ISO_NUM];        /* RW; Range: [0, 31]; Format:5.0; Sharpen Gain for Blue Area */
    hi_u8  skin_gain[HI_ISP_AUTO_ISO_NUM];      /* RW; Range: [0, 31]; Format:5.0;Sharpen Gain for Skin Area */
    hi_u16 max_sharp_gain[HI_ISP_AUTO_ISO_NUM]; /* RW; Range: [0, 0x7FF]; Format:8.3; Maximum sharpen gain */
} hi_isp_sharpen_auto_attr;

typedef struct {
    hi_bool enable;    /* RW; Range:[0, 1]; Format:1.0;Enable/Disable sharpen module */
    hi_u8 skin_umin;   /* RW; Range: [0, 255];  Format:8.0; U min value of the range of skin area */
    hi_u8 skin_vmin;   /* RW; Range: [0, 255];  Format:8.0; V min value of the range of skin area */
    hi_u8 skin_umax;   /* RW; Range: [0, 255];  Format:8.0; U max value of the range of skin area */
    hi_u8 skin_vmax;   /* RW; Range: [0, 255];  Format:8.0; V max value of the range of skin area */
    hi_op_mode op_type;
    hi_isp_sharpen_manual_attr manual_attr;
    hi_isp_sharpen_auto_attr   auto_attr;
} hi_isp_sharpen_attr;

typedef struct {
    hi_bool enable;     /* RW; Range:[0, 1]; Format:1.0;Enable/Disable Edge Mark */
    hi_u8   threshold;  /* RW; Range: [0, 255];  Format:8.0 */
    hi_u32  color;      /* RW; Range: [0, 0xFFFFFF];  Format:32.0 */
} hi_isp_edge_mark_attr;

/* High Light Constraint */
typedef struct {
    hi_bool enable;              /* RW; Range:[0, 1];  Format:1.0;Enable/Disable HLC module */
    hi_u8   luma_thr;            /* RW; Range:[0, 255];Format:8.0 */
    hi_u8   luma_target;         /* RW; Range:[0, 255];Format:8.0 */
} hi_isp_hlc_attr;

/* Crosstalk Removal */
typedef struct {
    hi_bool  enable;      /* RW; Range: [0, 1];Format 1.0;Enable/disable the crosstalk removal module */
    hi_u8    slope;       /* RW; Range:[0, 14];Crosstalk slope value. */
    hi_u8    sensi_slope;  /* RW; Range:[0, 14];Crosstalk sensitivity. */
    hi_u16   sensi_threshold; /* RW;Range:[0, 16383];Crosstalk sensitivity threshold. */
    hi_u16   strength[HI_ISP_AUTO_ISO_NUM];   /* RW; Range: [0,256];Crosstalk strength value. */
    hi_u16   threshold[HI_ISP_AUTO_ISO_NUM];  /* RW; Range:[0, 16383]; Crosstalk threshold. */
    hi_u16   np_offset[HI_ISP_AUTO_ISO_NUM];   /* RW; Range:[512, 16383]; Set Noise profile value. */
} hi_isp_cr_attr;

typedef struct {
    hi_u8    threshold;     /* RW;Range:[0x0,0x20];Format:6.0;Threshold for antifalsecolor */
    hi_u8    strength;      /* RW;Range:[0x0,0x1F];Format:5.0;Strength of antifalsecolor */
} hi_isp_anti_false_color_manual_attr;

typedef struct {
    hi_u8  threshold[HI_ISP_AUTO_ISO_NUM];    /* RW;Range:[0x0,0x20];Format:6.0;Threshold for antifalsecolor */
    hi_u8  strength[HI_ISP_AUTO_ISO_NUM];     /* RW;Range:[0x0,0x1F];Format:5.0;Strength of antifalsecolor */
} hi_isp_anti_false_color_auto_attr;

typedef struct {
    hi_bool  enable;                               /* RW;Range:[0x0,0x1];Format:1.0; AntiFalseColor Enable */
    hi_op_mode op_type;
    hi_isp_anti_false_color_auto_attr auto_attr;
    hi_isp_anti_false_color_manual_attr manual_attr;
} hi_isp_anti_false_color_attr;

typedef struct {
    hi_u8   nddm_strength;            /* RW; Range:[0x0, 0xFF]; Format:4.4; Non-direction strength */
    hi_u8   nddm_mf_detail_strength; /* RW; Range:[0x0, 0x7f];
                                        Format:3.4; Non-direction medium frequent detail enhance  strength */
    hi_u8   nddm_hf_detail_strength; /* RW; Range:[0x0, 0x10]; Format:2.2;
                                          Non-direction high frequent detail enhance strength */
    hi_u8   detail_smooth_range;    /* RW; Range:[0x1, 0x7]; Format:4.0; Detail smooth range */
    hi_u8   color_noise_f_threshold; /* RW;Range:[0x0,0xff]; Format:8.0; Threshold of color noise cancel */
    hi_u8   color_noise_f_strength; /* RW;Range:[0x0, 0x8]; Format:4.0; Strength of color noise cancel */
    hi_u8   color_noise_y_threshold; /* RW;Range: [0x0, 0xFF];
                                      Format:8.0; Range of color denoise luma, related to luminance and saturation */
    hi_u8   color_noise_y_strength; /* RW;Range:[0x0, 0xFF];
                                      Format:8.0;Format:8.0; Strength of color denoise luma */
} hi_isp_demosaic_manual_attr;

typedef struct {
    hi_u8   nddm_strength[HI_ISP_AUTO_ISO_NUM];     /* RW; Range:[0x0, 0xFF]; Format:4.4; Non-direction strength */
    hi_u8   nddm_mf_detail_strength[HI_ISP_AUTO_ISO_NUM]; /* RW; Range:[0x0, 0x7f]; Format:3.4; Non-direction medium
                                                             frequent detail enhance strength */
    hi_u8   nddm_hf_detail_strength[HI_ISP_AUTO_ISO_NUM]; /* RW; Range:[0x0, 0x10]; Format:2.2; Non-direction
                                                             high frequent detail enhance strength */
    hi_u8   detail_smooth_range[HI_ISP_AUTO_ISO_NUM];    /* RW; Range:[0x1, 0x7]; Format:4.0; Detail smooth range */
    hi_u8   color_noise_f_threshold[HI_ISP_AUTO_ISO_NUM]; /* RW;Range:[0x0, 0xff]; Format:8.0;
                                                             Threshold of color noise cancel */
    hi_u8   color_noise_f_strength[HI_ISP_AUTO_ISO_NUM]; /* RW;Range:[0x0, 0x8];
                                                            Format:4.0; Strength of color noise cancel */
    hi_u8   color_noise_y_threshold[HI_ISP_AUTO_ISO_NUM];  /* RW;Range:[0x0, 0xFF]
                                                              denoise luma, related to luminance and saturation */
    hi_u8   color_noise_y_strength[HI_ISP_AUTO_ISO_NUM];  /* RW;Range:[0x0, 0xFF]; Format:8.0;Format:8.0;
                                                                Strength of color denoise luma */
} hi_isp_demosaic_auto_attr;

typedef struct {
    hi_bool enable;        /* RW; Range:[0, 1]; Format:1.0;Enable/Disable demosaic module */
    hi_op_mode op_type;
    hi_isp_demosaic_manual_attr manual_attr;
    hi_isp_demosaic_auto_attr   auto_attr;
} hi_isp_demosaic_attr;

/* Defines the attributes of the ISP black level */
typedef struct {
    hi_op_mode     op_type;
    hi_u16         black_level[HI_ISP_BAYER_CHN_NUM]; /* RW; Range: [0x0, 0xFFF];Format:12.0;
                                                      Black level values that correspond to the black levels of the
                                                      R,Gr, Gb, and B components respectively. */
} hi_isp_black_level_attr;

/*
  Defines the FPN removal type
  0 = Frame mode correction
  1 = Line mode correction
 */
typedef enum {
    HI_ISP_FPN_TYPE_FRAME = 0,
    HI_ISP_FPN_TYPE_BUTT
} hi_isp_fpn_type;

/* Defines the information about calibrated black frames */
typedef struct {
    hi_u32              iso;             /* RW;Range:[0x64,0xFFFFFFFF];Format:32.0;FPN CALIBRATE ISO */
    hi_u32              offset[HI_ISP_STRIPING_MAX_NUM]; /* RW;Range:[0, 0xFFF];Format:12.0;
                                                          FPN frame u32Offset (agv pixel value) */
    hi_u32              frm_size;         /* RW;FPN frame size (exactly frm size or compress len) */
    hi_video_frame_info fpn_frame;         /* FPN frame info, 8bpp,10bpp,12bpp,16bpp. Compression or not */
} hi_isp_fpn_frame_info;

/* Defines the calibration attribute of the FPN removal module */
typedef struct {
    hi_u32                 threshold;  /* RW;Range:[1,0xFFF];Format:12.0;pix value > threshold means defective pixel */
    hi_u32                 frame_num;  /* RW;Range:[1, 16];Format:5.0;Number of frames to be calibrated.
                                        The value range is {1, 2, 4, 8, 16},that is, the integer exponentiation of 2 */
    hi_isp_fpn_type        fpn_type;   /* frame mode or line mode */
    hi_isp_fpn_frame_info  fpn_cali_frame;
} hi_isp_fpn_calibrate_attr;

/* Defines the manual correction attribute of the FPN removal module */
typedef struct {
    hi_u32  strength;         /* RW;Range:[0,1023];Format:10.0;Manual correction strength */
} hi_isp_fpn_manual_attr;

/* Defines the automatic correction attribute of the FPN removal module */
typedef struct {
    hi_u32 strength;          /* RW;Range:[0,1023];Format:10.0;Auto correction strength */
} hi_isp_fpn_auto_attr;

/* Defines the correction attribute of the FPN removal module */
typedef struct {
    hi_bool                enable;   /* RW;Range:[0,1];Format:1.0; */
    hi_op_mode         op_type;  /* manual or auto mode */
    hi_isp_fpn_type        fpn_type;
    hi_isp_fpn_frame_info  fpn_frm_info;
    hi_isp_fpn_manual_attr manual_attr;
    hi_isp_fpn_auto_attr   auto_attr;
} hi_isp_fpn_attr;

/* Defines the manual dehze attribute */
typedef struct {
    hi_u8 strength;  /* RW;Range:[0,0xFF];Format:8.0;Manual dehze strength */
} hi_isp_dehaze_manual_attr;

/* Defines the automatic dehze attribute */
typedef struct {
    hi_u8 strength; /* RW;Range:[0,0xFF];Format:8.0;Weighted coefficient for automatic dehaze strength. */
} hi_isp_dehaze_auto_attr;

/* Defines the ISP dehaze attribute */
typedef struct {
    hi_bool  enable;           /* RW;Range:[0,1];Format:1.0; */
    hi_bool  user_lut_enable;    /* RW;Range:[0,1];0:Auto Lut 1:User Lut */
    hi_u8    dehaze_lut[HI_ISP_DEHAZE_LUT_SIZE];
    hi_op_mode      op_type;
    hi_isp_dehaze_manual_attr manual_attr;
    hi_isp_dehaze_auto_attr   auto_attr;
    hi_u16       tmprflt_incr_coef; /* RW, Range: [0x0, 0x80].filter increase coeffcient. */
    hi_u16       tmprflt_decr_coef; /* RW, Range: [0x0, 0x80].filter decrease coeffcient. */
} hi_isp_dehaze_attr;

/* Defines purple fringing correction manual attribute */
typedef struct {
    hi_u8 de_purple_cr_str;   /* RW;Range: [0,8];Format:4.0;Correction strength of the R channel */
    hi_u8 de_purple_cb_str;   /* RW;Range: [0,8];Format:4.0;Correction strength of the B channel */
} hi_isp_depurplestr_manual_attr;

/* Defines purple fringing correction automatic attribute */
typedef struct {
    hi_u8 de_purple_cr_str[HI_ISP_LCAC_EXP_RATIO_NUM]; /* RW;Range: [0, 8];Format:4.0;
                                                           Correction strength of the R channel */
    hi_u8 de_purple_cb_str[HI_ISP_LCAC_EXP_RATIO_NUM]; /* RW;Range: [0, 8];Format:4.0;
                                                           Correction strength of the B channel */
} hi_isp_depurplestr_auto_attr;

/* Purple fringing detection and correction attribute */
typedef struct {
    hi_bool enable;               /* RW;Range: [0, 1];Format:1.0; enable/disable local cac */
    hi_u16  purple_detect_range;     /* RW;Range: [0, 410];Format:9.0;Purple fringing detection scope */
    hi_u16  var_threshold;              /* RW;Range: [0, 4095];Format:12.0;Edge detection threshold */
    hi_u16  r_detect_threshold[HI_ISP_LCAC_DET_NUM];   /* RW;Range: [0,4095];Format:12.0;Component
                                                          R detection threshold */
    hi_u16  g_detect_threshold[HI_ISP_LCAC_DET_NUM];   /* RW;Range: [0,4095];Format:12.0;Component
                                                          G detection threshold */
    hi_u16  b_detect_threshold[HI_ISP_LCAC_DET_NUM];   /* RW;Range: [0,4095];Format:12.0;Component
                                                          B detection threshold */
    hi_u16  luma_detect_threshold[HI_ISP_LCAC_DET_NUM];  /* RW;Range: [0,4095];Format:12.0;
                                                        Component Luma detection threshold */
    hi_s16  cb_cr_ratio[HI_ISP_LCAC_DET_NUM];  /* RW;Range: [-2048,2047];Format:S12.0; Cb/Cr ratio threshold */
    hi_op_mode      op_type;  /* RW;Range: [0, 1];Format:1.0;Purple fringing correction working mode */
    hi_isp_depurplestr_manual_attr manual_attr;
    hi_isp_depurplestr_auto_attr   auto_attr;
} hi_isp_local_cac_attr;

/* Defines the lateral chromatic aberration correction attribute */
typedef struct {
    hi_bool enable;            /* RW; Range: [0, 1];Format: 1.0;  enable/disable global cac */
    hi_u16  ver_coordinate;    /* RW; Range: [0, 8191];Format: 13.0; limited Range: [0, ImageHeight - 1],
                                  Vertical coordinate of the optical center */
    hi_u16  hor_coordinate;    /* RW; Range: [0, 8191];Format: 13.0; limited range : [0, ImageWidth - 1],
                                  Horizontal coordinate of the optical center */
    hi_s16  param_red_a;       /* RW; Range: [-256, 255];Format: 9.0;
                                  Coefficient a of the radius polynomial corresponding to channel R */
    hi_s16  param_red_b;       /* RW; Range: [-256, 255];Format: 9.0;
                                  Coefficient b of the radius polynomial corresponding to channel R */
    hi_s16  param_red_c;       /* RW; Range: [-256, 255];Format: 9.0;
                                  Coefficient c of the radius polynomial corresponding to channel R */
    hi_s16  param_blue_a;      /* RW; Range: [-256, 255];Format: 9.0;
                                  Coefficient a of the radius polynomial corresponding to channel B */
    hi_s16  param_blue_b;      /* RW; Range: [-256, 255];Format: 9.0;
                                  Coefficient b of the radius polynomial corresponding to channel B */
    hi_s16  param_blue_c;      /* RW; Range: [-256, 255];Format: 9.0;
                                  Coefficient c of the radius polynomial corresponding to channel B */
    hi_u8   ver_norm_shift;    /* RW; Range: [0, 7];Format: 3.0;
                                  Normalized shift parameter in the vertical direction */
    hi_u8   ver_norm_factor;   /* RW; Range: [0, 31];Format: 5.0;
                                  Normalized coefficient in the vertical direction */
    hi_u8   hor_norm_shift;    /* RW; Range: [0, 7];Format: 3.0;
                                  Normalized shift parameter in the horizontal direction */
    hi_u8   hor_norm_factor;   /* RW; Range: [0, 31];Format: 5.0; Normalized coefficient in the horizontal direction */
    hi_u16  cor_var_thr;       /* RW; Range: [0, 4095];Format: 12.0;
                                  Variance threshold of lateral chromatic aberration correction */
} hi_isp_global_cac_attr;

typedef struct {
    hi_u16 texture_strength[HI_ISP_SHARPEN_GAIN_NUM]; /* RW; range: [0, 4095]; Format:7.5;Undirectional
                                                         sharpen strength for texture and detail enhancement */
    hi_u16 edge_strength[HI_ISP_SHARPEN_GAIN_NUM]; /* RW; range: [0, 4095]; Format:7.5;Directional sharpen strength for
                                                  edge enhancement */
    hi_u16 texture_freq;         /* RW; range: [0, 4095];Format:6.6; Texture frequency adjustment.
                                    Texture and detail will be finer when it increase */
    hi_u16 edge_freq;            /* RW; range: [0, 4095];Format:6.6; Edge frequency adjustment.
                                    Edge will be narrower and thiner when it increase */
    hi_u8  over_shoot;           /* RW; range: [0, 127]; Format:7.0;u8OvershootAmt */
    hi_u8  under_shoot;          /* RW; range: [0, 127]; Format:7.0;u8UndershootAmt */
    hi_u8  shoot_sup_strength;        /* RW; range: [0, 255]; Format:8.0;overshoot and undershoot suppression strength,
                                    the amplitude and width of shoot will be decrease when shootSupSt increase */
    hi_u16  coarse_strength[HI_ISP_BAYER_CHN_NUM];    /* RW;Range:[0x0, 0x3ff];Format:10.0;
                                                 Coarse Strength of noise reduction */
    hi_u8   wdr_frame_strength[HI_ISP_WDR_MAX_FRAME_NUM]; /* RW;Range:[0x0, 0x50];Format:7.0;
                                                 Coarse strength of each frame in wdr mode */
    hi_u8   chroma_strength[HI_ISP_BAYER_CHN_NUM];  /* Not Support */
    hi_u8   fine_strength;             /* RW;Range:[0x0,0x80];Format:8.0;Strength of Luma noise reduction */
    hi_u16  coring_wgt;           /* RW;Range:[0x0,0xC80];Format:12.0;Strength of reserving the random noise */

    hi_u16 dehaze_strength_actual; /* RW;Range:[0,0xFF];Format:8.0;actual dehze strength */
    hi_u16 drc_strength_actual;  /* RW;Range: [0x0, 0x3FF]; Strength of dynamic range compression.
                                    Higher values lead to higher differential gain between shadows and highlights. */
    hi_u32 wdr_exp_ratio_actual[HI_ISP_WDR_MAX_FRAME_NUM - 1]; /* RW; Range:[0x40, 0x4000]; Format:26.6; 0x40 means
                                                                1 times. When enExpRatioType is HI_OP_MODE_AUTO,
                                                                u32ExpRatio is invalid.
                                                                When enExpRatioType is HI_OP_MODE_MANUAL, u32ExpRatio is
                                                                quotient of long exposure time / short exposure time. */
    hi_bool wdr_switch_finish;       /* RW; Range:[0, 1];Format:1.0;HI_TRUE: WDR switch is finished */
    hi_bool res_switch_finish;       /* RW; Range:[0, 1];Format:1.0;HI_TRUE: Resolution switch is finished */
    hi_u16  black_level_actual[HI_ISP_BAYER_CHN_NUM]; /* RW; Range: [0x0, 0xFFF];Format:12.0;
                                             Actual Black level values that correspond to the black levels of
                                             the R,Gr, Gb, and B components respectively. */
} hi_isp_inner_state_info;

/*
  The position of AE histogram in ISP BE pipeline
  0 = After DG;
  1 = After static WB;
  2 = After DRC;
 */
typedef enum {
    HI_ISP_AE_AFTER_DG = 0,
    HI_ISP_AE_AFTER_WB,
    HI_ISP_AE_AFTER_DRC,
    HI_ISP_AE_SWITCH_BUTT
} hi_isp_ae_switch;

/*
  FourPlaneMode enable
  0 : Disabled;
  1 : Enabled
*/
typedef enum {
    HI_ISP_AE_FOUR_PLANE_MODE_DISABLE = 0,
    HI_ISP_AE_FOUR_PLANE_MODE_ENABLE,
    HI_ISP_AE_FOUR_PLANE_MODE_BUTT
} hi_isp_ae_four_plane_mode;

typedef enum {
    HI_ISP_AE_HIST_SKIP_EVERY_PIXEL = 0,
    HI_ISP_AE_HIST_SKIP_EVERY_2ND_PIXEL,
    HI_ISP_AE_HIST_SKIP_EVERY_3RD_PIXEL,
    HI_ISP_AE_HIST_SKIP_EVERY_4TH_PIXEL,
    HI_ISP_AE_HIST_SKIP_EVERY_5TH_PIXEL,
    HI_ISP_AE_HIST_SKIP_EVERY_8TH_PIXEL,
    HI_ISP_AE_HIST_SKIP_EVERY_9TH_PIXEL,
    HI_ISP_AE_HIST_SKIP_BUTT
} hi_isp_ae_hist_skip;

typedef enum {
    HI_ISP_AE_HIST_START_FORM_FIRST_COLUMN = 0,
    HI_ISP_AE_HIST_START_FORM_SECOND_COLUMN,
    HI_ISP_AE_HIST_OFFSET_X_BUTT
} hi_isp_ae_hist_offset_x;

typedef enum {
    HI_ISP_AE_HIST_START_FORM_FIRST_ROW = 0,
    HI_ISP_AE_HIST_START_FORM_SECOND_ROW,
    HI_ISP_AE_HIST_OFFSET_Y_BUTT
} hi_isp_ae_hist_offset_y;

/* Defines the mode configuration for the sampling points during global histogram statistics. */
typedef struct {
    hi_isp_ae_hist_skip hist_skip_x;  /* RW; Range:[0, 6]; Format:4.0;
                                         Histogram decimation in horizontal direction: 0=every pixel;
                                         1=every 2nd pixel; 2=every 3rd pixel; 3=every 4th pixel; 4=every 5th pixel;
                                         5=every 8th pixel; 6+=every 9th pixel */
    hi_isp_ae_hist_skip hist_skip_y;  /* RW; Range:[0, 6]; Format:4.0;
                                         Histogram decimation in vertical direction: 0=every pixel; 1=every 2nd pixel;
                                         2=every 3rd pixel; 3=every 4th pixel; 4=every 5th pixel; 5=every 8th pixel;
                                         6+=every 9th pixel */
    hi_isp_ae_hist_offset_x hist_offset_x;  /* RW; Range:[0, 1]; Format:1.0;
                                               0= start from the first column; 1=start from second column */
    hi_isp_ae_hist_offset_y hist_offset_y;  /* RW; Range:[0, 1]; Format:1.0;
                                               0= start from the first row; 1= start from second row */
} hi_isp_ae_hist_config;

typedef enum {
    HI_ISP_AE_NORMAL = 0,
    HI_ISP_AE_ROOT,
    HI_ISP_AE_STAT_MODE_BUTT
} hi_isp_ae_stat_mode;

/* Crops the input image of the AE module */
typedef struct {
    hi_bool enable; /* RW; Range: [0,1];  Format:1.0;AE crop enable. */
    hi_u16  x;   /* RW; Range: [0, 4608 - 256];
                    Format:13.0;AE image crop start x, limited range:[0, ImageWidth - 256] */
    hi_u16  y;   /* RW; Range: [0, 4608 - 120];
                    Format:13.0;AEimage crop start y, limited range:[0, ImageHeight - 120]  */
    hi_u16  w;   /* RW; Range: [256, 4608];
                    Format:14.0;AE image crop width,  limited range:[256, ImageWidth] */
    hi_u16  h;   /* RW; Range: [120, 4608];
                    Format:14.0;AE image crop height  limited range:[120, ImageHeight] */
} hi_isp_ae_crop;

/* config of statistics structs */
typedef struct {
    hi_isp_ae_switch          ae_switch;          /* RW; Range:[0, 2]; Format:2.0;
                                                     The position of AE histogram in ISP BE pipeline */
    hi_isp_ae_hist_config     hist_config;
    hi_isp_ae_four_plane_mode four_plane_mode;    /* RW; Range:[0, 1]; Format:2.0;Four Plane Mode Enable */
    hi_isp_ae_stat_mode       hist_mode;          /* RW; Range:[0, 1]; Format:2.0;AE Hist Rooting Mode */
    hi_isp_ae_stat_mode       aver_mode;          /* RW; Range:[0, 1]; Format:2.0;AE Aver Rooting Mode */
    hi_isp_ae_stat_mode       max_gain_mode;      /* RW; Range:[0, 1]; Format:2.0;Max Gain Rooting Mode */
    hi_isp_ae_crop            crop;
    hi_u8 weight[HI_ISP_AE_ZONE_ROW][HI_ISP_AE_ZONE_COLUMN]; /* RW; Range:[0x0, 0xF]; Format:4.0; AE weighting table */
} hi_isp_ae_stats_cfg;

/* fines whether the peak value of the zoned IIR statistics is calculated.  */
typedef enum {
    HI_ISP_AF_STA_NORM         = 0,    /* use every value of the block statistic */
    HI_ISP_AF_STA_PEAK,                /* use the maximum value in one line of the block statistic */
    HI_ISP_AF_STA_BUTT
} hi_isp_af_peak_mode;

/* Defines whether the zoned statistics are squared in square mode.  */
typedef enum {
    HI_ISP_AF_STA_SUM_NORM     = 0,    /* statistic value accumulate */
    HI_ISP_AF_STA_SUM_SQU,             /* statistic value square then accumulate */
    HI_ISP_AF_STA_SUM_BUTT
}  hi_isp_af_squ_mode;

/* Crops the input image of the AF module */
typedef struct {
    hi_bool enable; /* RW; Range: [0,1];  Format:1.0; AF crop enable */
    hi_u16  x;    /* RW; Range: [0, 4352];
                     Format:13.0;AF image crop start x, limited range:[0, ImageWidth-256] */
    hi_u16  y;    /* RW; Range: [0, 4488];
                     Format:13.0;AF image crop start y, limited range:[0, ImageHeight-120] */
    hi_u16  w;    /* RW; Range: [256, 4608];
                     Format:14.0;AF image crop width, limited range:[256, ImageWidth] */
    hi_u16  h;    /* RW; Range: [120, 4608];
                     Format:14.0;AF image crop height, limited range:[120, ImageHeight] */
} hi_isp_af_crop;

/* Defines the position of AF module statistics.  */
typedef enum {
    HI_ISP_AF_STATS_AFTER_DGAIN = 0, /* The AF module is placed in the raw field for statistics,AF after DGain */
    HI_ISP_AF_STATS_AFTER_DRC,         /* The AF module is placed in the raw field for statistics,AF after DRC */
    HI_ISP_AF_STATS_AFTER_CSC,         /* The AF module is placed in the YUV field for statistics,AF after CSC */
    HI_ISP_AF_STATS_BUTT
} hi_isp_af_stats_pos;

/* Configures the Bayer field of the AF module */
typedef struct {
    hi_u8               gamma_gain_limit; /* RW; Range: [0x0, 0x5]; Format:3.0 */
    hi_u8               gamma_value;      /* RW; Range: [0x0, 0x6] */
    hi_isp_bayer_format pattern;          /* RW; Range: [0x0, 0x3]; Format:2.0;raw domain pattern */
} hi_isp_af_raw_cfg;

/* Configures the pre-filter of the AF module */
typedef struct {
    hi_bool             en;         /* RW; Range: [0,1]; Format:1.0; pre filter enable  .          */
    hi_u16              strength; /* RW; Range: [0x0, 0xFFFF]; Format:16.0;pre filter strength    */
} hi_isp_af_pre_filter_cfg;

/* Defines AF statistics configuration parameters */
typedef struct {
    hi_bool                 enable;     /* RW; Range: [0,1];   AF enable. */
    hi_u16                  h_wnd;      /* RW; Range: [1, 17]; AF statistic window horizontal block.  */
    hi_u16                  v_wnd;      /* RW; Range: [1, 15]; AF statistic window vertical block.  */
    hi_u16                  h_size;     /* RW; Range: [256, 4608] */
    hi_u16                  v_size;    /* RW; Range: [120, 4608];
                                          limited Range: [120, ImageHeight],AF image height. */
    hi_isp_af_peak_mode      peak_mode;      /* RW; Range: [0,1]; AF peak value statistic mode. */
    hi_isp_af_squ_mode       squ_mode;       /* RW; Range: [0,1]; AF statistic square accumulate. */
    hi_isp_af_crop           crop;           /* RW; AF input image crop */
    hi_isp_af_stats_pos      stats_pos; /* RW; Range: [0,2]; AF statistic position, it can be set to yuv or raw */
    hi_isp_af_raw_cfg        raw_cfg;        /* RW; When AF locate at RAW domain, these para should be cfg. */
    hi_isp_af_pre_filter_cfg pre_flt_cfg;    /* RW; pre filter cfg */
    hi_u16                   high_luma_th;   /* RW; Range: [0,0xFF]; high luma threshold. */
} hi_isp_af_cfg;

/* Configures the AF level depend gain module */
typedef struct {
    hi_bool     ld_en;                    /* RW; Range: [0, 1]; FILTER level depend gain enable. */
    hi_u16      th_low;                   /* RW; range: [0x0, 0xFF]; FILTER level depend th low */
    hi_u16      gain_low;                 /* RW; range: [0x0, 0xFF]; FILTER level depend gain low */
    hi_u16      slp_low;                  /* RW; range: [0x0, 0xF];  FILTER level depend slope low */
    hi_u16      th_high;                  /* RW; range: [0x0, 0xFF]; FILTER level depend th high */
    hi_u16      gain_high;                /* RW; range: [0x0, 0xFF]; FILTER level depend gain high */
    hi_u16      slp_high;                 /* RW; range: [0x0, 0xF];  FILTER level depend slope high */
} hi_isp_af_ld;

/* Configures the AF coring module */
typedef struct {
    hi_u16      th;                      /* RW; Range: [0x0, 0x7FF];FILTER coring threshold. */
    hi_u16      slp;                     /* RW; Range: [0x0, 0xF];  FILTER Coring Slope */
    hi_u16      lmt;                     /* RW; Range: [0x0, 0x7FF];FILTER coring limit */
} hi_isp_af_coring;

#define HI_ISP_IIR_EN_NUM                                3
#define HI_ISP_IIR_GAIN_NUM                              7
#define HI_ISP_IIR_SHIFT_NUM                             4
/* Defines the IIR parameter configuration of horizontal filters for AF statistics. */
typedef struct {
    hi_bool          narrow_band;             /* RW; Range: [0, 1]; IIR narrow band enable. */
    hi_bool          iir_en[HI_ISP_IIR_EN_NUM];      /* RW; Range: [0, 1]; IIR enable. */
    hi_u8            iir_shift;               /* Not Support */
    hi_s16           iir_gain[HI_ISP_IIR_GAIN_NUM];  /* RW; Range: [-511, 511]. IIR gain,gain0 range:[0,255]. */
    hi_u16           iir_shift_lut[HI_ISP_IIR_SHIFT_NUM]; /* RW; Range: [0x0, 0x7];  IIR shift. */
    hi_isp_af_ld     ld;                       /* RW; filter level depend. */
    hi_isp_af_coring coring;                   /* RW; filter coring. */
} hi_isp_af_h_param;

#define HI_ISP_FIR_GAIN_NUM                              5
typedef struct {
    hi_s16           firh[HI_ISP_FIR_GAIN_NUM]; /* RW; Range: [-31, 31];   FIR gain. */
    hi_isp_af_ld     ld;                 /* RW; filter level depend. */
    hi_isp_af_coring coring;             /* RW; filter coring.  */
} hi_isp_af_v_param;


#define HI_ISP_ACC_SHIFT_H_NUM                           2
#define HI_ISP_ACC_SHIFT_V_NUM                           2
typedef struct {
    hi_u16 acc_shift_y;                   /* RW; Range: [0x0, 0xF]; luminance Y statistic shift. */
    hi_u16 acc_shift_h[HI_ISP_ACC_SHIFT_H_NUM];  /* RW; Range: [0x0, 0xF]; IIR statistic shift. */
    hi_u16 acc_shift_v[HI_ISP_ACC_SHIFT_V_NUM];  /* RW; Range: [0x0, 0xF]; FIR statistic shift. */
    hi_u16 hl_cnt_shift;                  /* RW; Range: [0x0, 0xF]; High luminance counter shift */
} hi_isp_af_fv_param;

typedef struct {
    hi_isp_af_cfg        config;
    hi_isp_af_h_param    h_param_iir0;
    hi_isp_af_h_param    h_param_iir1;
    hi_isp_af_v_param    v_param_fir0;
    hi_isp_af_v_param    v_param_fir1;
    hi_isp_af_fv_param   fv_param;
} hi_isp_focus_stats_cfg;

/*
   the main purpose of stat key was to access individual statistic info separately...
   ...for achieving performance optimization of CPU, because of we acquire stat...
   ... in ISP_DRV ISR for our firmware reference and USER can also use alternative MPI...
   ... to do this job, so bit1AeStat1~bit1AfStat for MPI behavior control, and bit16IsrAccess...
   ... for ISR access control, they were independent. but they have the same bit order, for example...
   ... bit1AeStat1 for MPI AeStat1 access key, and bit16 of u32Key for ISR AeStat1 access key
 */
typedef union {
    hi_u64  key;
    struct {
        hi_u64  bit1_fe_ae_global_stat      : 1;   /* [0] */
        hi_u64  bit1_fe_ae_local_stat      : 1;   /* [1] */
        hi_u64  bit1_fe_ae_stitch_global_stat  : 1;   /* [2] Not Support */
        hi_u64  bit1_fe_ae_stitch_local_stat  : 1;   /* [3] Not Support */
        hi_u64  bit1_be_ae_global_stat      : 1;   /* [4] */
        hi_u64  bit1_be_ae_local_stat      : 1;   /* [5] */
        hi_u64  bit1_be_ae_stitch_global_stat  : 1;   /* [6] Not Support */
        hi_u64  bit1_be_ae_stitch_local_stat  : 1;   /* [7] Not Support */
        hi_u64  bit1_awb_stat1           : 1;   /* [8] */
        hi_u64  bit1_awb_stat2           : 1;   /* [9] */
        hi_u64  bit2_reserved0           : 2;   /* [10:11] */
        hi_u64  bit1_fe_af_stat          : 1;   /* [12] Not Support */
        hi_u64  bit1_be_af_stat          : 1;   /* [13] */
        hi_u64  bit2_reserved1           : 2;   /* [14:15] */
        hi_u64  bit1_dehaze              : 1;   /* [16] */
        hi_u64  bit1_mg_stat             : 1;   /* [17] */
        hi_u64  bit14_reserved           : 14;   /* [18:31] */
        hi_u64  bit32_isr_access         : 32;   /* [32:63] */
    };
} hi_isp_stats_ctrl;

/* statistics structs */
#define HI_ISP_HIST_NUM                                  1024
#define HI_ISP_BAYER_PATTERN_NUM                         4
#define HI_ISP_WDR_CHN_MAX                               4
#define HI_ISP_CHN_MAX_NUM                               4
typedef struct {
    hi_u16 grid_y_pos[HI_ISP_AE_ZONE_ROW + 1];   /* R */
    hi_u16 grid_x_pos[HI_ISP_AE_ZONE_COLUMN + 1]; /* R */
    hi_u8  status;                        /* R;0:not update, 1: update,others:reserved */
} hi_isp_ae_grid_info;

typedef struct {
    hi_u16 grid_y_pos[HI_ISP_MG_ZONE_ROW + 1];    /* R */
    hi_u16 grid_x_pos[HI_ISP_MG_ZONE_COLUMN + 1]; /* R */
    hi_u8  status;                         /* R;0:not update, 1: update,others:reserved */
} hi_isp_mg_grid_info;

typedef struct {
    hi_u16 grid_y_pos[HI_ISP_AWB_ZONE_ORIG_ROW  + 1];   /* R */
    hi_u16 grid_x_pos[HI_ISP_AWB_ZONE_ORIG_COLUMN + 1]; /* R */
    hi_u8  status;                               /* R;0:not update, 1: update,others:reserved */
} hi_isp_awb_grid_info;

typedef struct {
    hi_u16 grid_y_pos[HI_ISP_AF_ZONE_ROW + 1];    /* R */
    hi_u16 grid_x_pos[HI_ISP_AF_ZONE_COLUMN + 1]; /* R */
    hi_u8  status;                         /* R;0:not update, 1: update,others:reserved */
} hi_isp_focus_grid_info;

typedef struct {
    hi_u32 fe_hist1024_value[HI_ISP_CHN_MAX_NUM][HI_ISP_HIST_NUM];  /* R; Range: [0x0, 0xFFFFFFFF]; Format:32.0;
                                                             Global 1024 bins histogram of FE */
    hi_u16 fe_global_avg[HI_ISP_CHN_MAX_NUM][HI_ISP_BAYER_PATTERN_NUM]; /* R; Range: [0x0, 0xFFFF]; Format:16.0;
                                                                 Global average value of FE */
    hi_u16 fe_zone_avg[HI_ISP_CHN_MAX_NUM][HI_ISP_AE_ZONE_ROW]
        [HI_ISP_AE_ZONE_COLUMN][HI_ISP_BAYER_PATTERN_NUM];    /* Not Support */
    hi_u32 be_hist1024_value[HI_ISP_HIST_NUM];                /* R; Range: [0x0, 0xFFFFFFFF]; Format:32.0;
                                                                           Global 1024 bins histogram of BE */
    hi_u16 be_global_avg[HI_ISP_BAYER_PATTERN_NUM];           /* R; Range: [0x0, 0xFFFF]; Format:16.0;
                                                                           Global average value of BE */
    hi_u16 be_zone_avg[HI_ISP_AE_ZONE_ROW][HI_ISP_AE_ZONE_COLUMN][HI_ISP_BAYER_PATTERN_NUM]; /* R;
                                                                           Range: [0x0, 0xFFFF]; Format:16.0;
                                                                           Zone average value of BE */
    hi_isp_ae_grid_info fe_grid_info;
    hi_isp_ae_grid_info be_grid_info;
} hi_isp_ae_stats;

typedef struct {
    hi_u16 au16_zone_avg[HI_ISP_MG_ZONE_ROW][HI_ISP_MG_ZONE_COLUMN][HI_ISP_BAYER_PATTERN_NUM]; /* R;
                                                                             Range: [0x0, 0xFF]; Format:8.0;
                                                                             Zone average value */
    hi_isp_mg_grid_info grid_info;
} hi_isp_mg_stats;

/* AWB */
typedef enum {
    HI_ISP_AWB_AFTER_DG                   = 0,
    HI_ISP_AWB_AFTER_Expander,
    HI_ISP_AWB_AFTER_DRC,
    HI_ISP_AWB_SWITCH_BUTT
} hi_isp_awb_switch;

/* Crops the input image of the AWB module */
typedef struct {
    hi_bool enable;  /* RW; Range: [0,1];  Format:1.0;AWB crop enable */
    hi_u16  x;    /* RW; Range: [0, 4608 - 60];
                     Format:13.0;AWB image crop start x, limited range:[0, ImageWidth - u16ZoneCol * 60] */
    hi_u16  y;    /* RW; Range: [0, 4608 - 14];
                     Format:13.0;AWB image crop start y,limited range:[0, ImageHeight - u16ZoneRow * 14] */
    hi_u16  w;    /* RW; Range: [60, 4608];
                     Format:14.0; AWB image crop width, limited range:[u16ZoneCol * 60, ImageWidth] */
    hi_u16  h;    /* RW; Range: [14, 4608];
                     Format:14.0; AWB image crop height, limited range:[u16ZoneRow * 14, ImageHeight] */
} hi_isp_awb_crop;

/* Defines the AWB statistics configuration */
typedef struct {
    hi_isp_awb_switch awb_switch; /* RW; Range: [0x0, 0x2]; Position of AWB statistics in pipeline */
    hi_u16 zone_row;           /* RW; Range: [0x1, 0x20]; Vertical Blocks,
                                  limited range:[1, min(32, ImageHeight /AWB_MIN_HEIGHT)] */
    hi_u16 zone_col;           /* RW; Range: [0x1, 0x20]; Horizontal Blocks,
                                  limited range:[BlkNum, min(32, Width /AWB_MIN_WIDTH)] */
    hi_u16 zone_bin;           /* RW; Range:[1, 1], Brightness Bins */
    hi_u16 hist_bin_thresh[HI_ISP_AWB_ZONE_BIN_MAX]; /* RW; Range: [0x0, 0xFFFF]; Bin Threshold */
    hi_u16 white_level;        /* RW; Range: [0x0, 0xFFFF];Upper limit of valid data for white region,
                                  for Bayer statistics, [0x0, 0x3FF] for RGB statistics */
    hi_u16 black_level;        /* RW; Range: [0x0, 0xFFFF];limited range: [0x0, u16WhiteLevel],
                                  Lower limit of valid data for white region .
                                  for Bayer statistics, bitwidth is 12, for RGB statistics, bitwidth is 10 */
    hi_u16 cb_max;             /* RW; Range: [0x0, 0xFFF];Maximum value of B/G for white region */
    hi_u16 cb_min;             /* RW; Range: [0x0, 0xFFF];
                                  limited range: [0x0, u16CbMax]Minimum value of B/G for white region */
    hi_u16 cr_max;             /* RW; Range: [0x0, 0xFFF];Maximum value of R/G for white region */
    hi_u16 cr_min;             /* RW; Range: [0x0, 0xFFF];
                                  limited range: [0x0, u16CrMax],Minimum value of R/G for white region */
    hi_isp_awb_crop crop;
} hi_isp_wb_stats_cfg;

typedef struct {
    hi_u16  zone_row;                            /* R; Range: [0x0, 0x20];
                                                    effective horizontal block number for AWB statistic stitch window */
    hi_u16  zone_col;                            /* R; Range: [0x0, 0x80];
                                                    effective vertical block number for AWB statistic stitch window */
    hi_u16  zone_avg_r[HI_ISP_AWB_ZONE_STITCH_MAX];     /* R; Range: [0x0, 0xFFFF];Zone Average R  for Stitch mode */
    hi_u16  zone_avg_g[HI_ISP_AWB_ZONE_STITCH_MAX];     /* R; Range: [0x0, 0xFFFF];Zone Average G  for Stitch mode */
    hi_u16  zone_avg_b[HI_ISP_AWB_ZONE_STITCH_MAX];     /* R; Range: [0x0, 0xFFFF];Zone Average B  for Stitch mode */
    hi_u16  zone_count_all[HI_ISP_AWB_ZONE_STITCH_MAX]; /* R; Range: [0x0, 0xFFFF];
                                                           normalized number of Gray points  for Stitch mode */
} hi_isp_wb_stitch_stats;

typedef struct {
    hi_u16 global_r;          /* R; Range: [0x0, 0xFFFF];Global WB output Average R */
    hi_u16 global_g;          /* R; Range: [0x0, 0xFFFF];Global WB output Average G */
    hi_u16 global_b;          /* R; Range: [0x0, 0xFFFF];Global WB output Average B */
    hi_u16 count_all;         /* R; Range: [0x0, 0xFFFF];normalized number of Gray points */

    hi_u16 zone_avg_r[HI_ISP_AWB_ZONE_NUM];            /* R; Range: [0x0, 0xFFFF];Zone Average R */
    hi_u16 zone_avg_g[HI_ISP_AWB_ZONE_NUM];            /* R; Range: [0x0, 0xFFFF];Zone Average G */
    hi_u16 zone_avg_b[HI_ISP_AWB_ZONE_NUM];            /* R; Range: [0x0, 0xFFFF];Zone Average B */
    hi_u16 zone_count_all[HI_ISP_AWB_ZONE_NUM];        /* R; Range: [0x0, 0xFFFF];normalized number of Gray points */
    hi_isp_awb_grid_info grid_info;
} hi_isp_wb_stats;

typedef struct {
    hi_u16  v1;            /* R; Range: [0x0, 0xFFFF];vertical fir block1 Statistics */
    hi_u16  h1;            /* R; Range: [0x0, 0xFFFF];horizontal iir block1 Statistics */
    hi_u16  v2;            /* R; Range: [0x0, 0xFFFF];vertical fir block2 Statistics */
    hi_u16  h2;            /* R; Range: [0x0, 0xFFFF];horizontal iir block2 Statistics */
    hi_u16  y;             /* R; Range: [0x0, 0xFFFF];Y Statistics */
    hi_u16  hl_cnt;        /* R; Range: [0x0, 0xFFFF];HlCnt Statistics */
} hi_isp_focus_zone;

typedef struct {
    /* R; The zoned measure of contrast */
    hi_isp_focus_zone zone_metrics[HI_ISP_WDR_CHN_MAX][HI_ISP_AF_ZONE_ROW][HI_ISP_AF_ZONE_COLUMN];
} hi_isp_fe_focus_stats;

typedef struct {
    hi_isp_focus_zone zone_metrics[HI_ISP_AF_ZONE_ROW][HI_ISP_AF_ZONE_COLUMN]; /* R; The zoned measure of contrast */
} hi_isp_be_focus_stats;

typedef struct {
    hi_isp_fe_focus_stats  fe_af_stat; /* Not Support */
    hi_isp_be_focus_stats  be_af_stat;
    hi_isp_focus_grid_info      fe_af_grid_info; /* Not Support */
    hi_isp_focus_grid_info      be_af_grid_info;
} hi_isp_af_stats;

typedef struct {
    hi_isp_stats_ctrl      key;
    hi_isp_ae_stats_cfg    ae_cfg;
    hi_isp_wb_stats_cfg    wb_cfg;
    hi_isp_focus_stats_cfg focus_cfg;
} hi_isp_stats_cfg;

/* ISP debug information */
typedef struct {
    hi_bool debug_en;     /* RW; 1:enable debug, 0:disable debug */
    hi_u64  phy_addr;     /* RW; phy address of debug info */
    hi_u32  depth;        /* RW; depth of debug info */
} hi_isp_debug_info;
/*
0 = Communication between the sensor and the ISP over the I2C interface
1 = Communication between the sensor and the ISP over the SSP interface
*/
typedef enum {
    HI_ISP_SNS_I2C_TYPE = 0,
    HI_ISP_SNS_SSP_TYPE,

    HI_ISP_SNS_TYPE_BUTT,
} hi_isp_sns_type;

/* sensor communication bus */
typedef union {
    hi_s8   i2c_dev;
    struct {
        hi_s8  bit4_ssp_dev       : 4;
        hi_s8  bit4_ssp_cs        : 4;
    } ssp_dev;
} hi_isp_sns_commbus;

typedef struct {
    hi_bool update;        /* RW; Range: [0x0, 0x1]; Format:1.0;
                              HI_TRUE: The sensor registers are written,
                              HI_FALSE: The sensor registers are not written */
    hi_u8   delay_frm_num; /* RW; Number of delayed frames for the sensor register */
    hi_u8   int_pos;       /* RW;Position where the configuration of the sensor register takes effect */
    hi_u8   dev_addr;      /* RW;Sensor device address */
    hi_u32  reg_addr;      /* RW;Sensor register address */
    hi_u32  addr_byte_num; /* RW;Bit width of the sensor register address */
    hi_u32  data;          /* RW;Sensor register data */
    hi_u32  data_byte_num; /* RW;Bit width of sensor register data */
} hi_isp_i2c_data;

typedef struct {
    hi_bool update;             /* RW; Range: [0x0, 0x1]; Format:1.0;
                                   HI_TRUE: The sensor registers are written,
                                   HI_FALSE: The sensor registers are not written */
    hi_u8   delay_frm_num;      /* RW; Number of delayed frames for the sensor register */
    hi_u8   int_pos;            /* RW;Position where the configuration of the sensor register takes effect */
    hi_u32  dev_addr;           /* RW;Sensor device address */
    hi_u32  dev_addr_byte_num;  /* RW;Bit width of the sensor device address */
    hi_u32  reg_addr;           /* RW;Sensor register address */
    hi_u32  reg_addr_byte_num;  /* RW;Bit width of the sensor register address */
    hi_u32  data;               /* RW;Sensor register data */
    hi_u32  data_byte_num;      /* RW;Bit width of sensor register data */
} hi_isp_ssp_data;

typedef struct {
    hi_isp_sns_type sns_type;
    hi_u32  reg_num;               /* RW;Number of registers required when exposure results are written to the sensor.
                                      The member value cannot be dynamically changed */
    hi_u8   cfg2_valid_delay_max;  /* RW;Maximum number of delayed frames from the time when all sensor registers are
                                      configured to the time when configurations take effect, which is used to ensure
                                      the synchronization between sensor registers and ISP registers */
    hi_isp_sns_commbus  com_bus;
    union {
        hi_isp_i2c_data i2c_data[HI_ISP_MAX_SNS_REGS];
        hi_isp_ssp_data ssp_data[HI_ISP_MAX_SNS_REGS];
    };

    struct {
        hi_bool update;
        hi_u8   delay_frm_num;
        hi_u32  slave_vs_time;      /* RW;time of vsync. Unit: inck clock cycle */
        hi_u32  slave_bind_dev;
    } slv_sync;

    hi_bool config;
} hi_isp_sns_regs_info;

typedef enum {
    HI_ISP_VD_FE_START   = 0,
    HI_ISP_VD_FE_END,
    HI_ISP_VD_BE_END,

    HI_ISP_VD_BUTT
} hi_isp_vd_type;

/* Defines the attributes of the virtual addresses for the registers of ISP submodules */
typedef struct {
    hi_void *isp_ext_reg_addr;        /* R;Start virtual address for the ISP external virtual registers */
    hi_u32  isp_ext_reg_size;         /* R;Size of the ISP external virtual registers */
    hi_void *ae_ext_reg_addr;         /* R;Start virtual address for the AE library module */
    hi_u32  ae_ext_reg_size;          /* R;Size of the AE library module */
    hi_void *awb_ext_reg_addr;        /* R;Start virtual address for the AWB library module */
    hi_u32  awb_ext_reg_size;         /* R;Size of the AWB library module */
} hi_isp_reg_attr;

/* AI structs */
/*
  Defines the ISP iris type
  0 = fixed iris
  1 = DC iris
  2 = P iris
 */
typedef enum {
    HI_ISP_IRIS_FIXED_TYPE = 0,
    HI_ISP_IRIS_DC_TYPE = 1,
    HI_ISP_IRIS_P_TYPE = 2,
    HI_ISP_IRIS_TYPE_BUTT,
} hi_isp_iris_type;

/* Defines the F value of the ISP iris */
typedef enum {
    HI_ISP_IRIS_F_NO_32_0 = 0,
    HI_ISP_IRIS_F_NO_22_0,
    HI_ISP_IRIS_F_NO_16_0,
    HI_ISP_IRIS_F_NO_11_0,
    HI_ISP_IRIS_F_NO_8_0,
    HI_ISP_IRIS_F_NO_5_6,
    HI_ISP_IRIS_F_NO_4_0,
    HI_ISP_IRIS_F_NO_2_8,
    HI_ISP_IRIS_F_NO_2_0,
    HI_ISP_IRIS_F_NO_1_4,
    HI_ISP_IRIS_F_NO_1_0,

    HI_ISP_IRIS_F_NO_BUTT,
} hi_isp_iris_f_no;

typedef struct {
    hi_s32 kp;             /* RW; Range:[0, 100000]; Format:32.0; the proportional gain of PID algorithm,
                              default value is 7000 */
    hi_s32 ki;             /* RW; Range:[0, 1000]; Format:32.0; the integral gain of PID algorithm,
                              default value is 100 */
    hi_s32 kd;             /* RW; Range:[0, 100000]; Format:32.0; the derivative gain of PID algorithm,
                              default value is 3000 */
    hi_u32 min_pwm_duty;   /* RW; Range:[0, 1000]; Format:32.0; which is the min pwm duty for dciris control */
    hi_u32 max_pwm_duty;   /* RW; Range:[0, 1000]; Format:32.0; which is the max pwm duty for dciris control */
    hi_u32 open_pwm_duty;  /* RW; Range:[0, 1000]; Format:32.0; which is the open pwm duty for dciris control */
} hi_isp_dciris_attr;

typedef struct {
    hi_bool step_fno_table_change;    /* W; Range:[0, 1]; Format:1.0; Step-F number mapping table change or not */
    hi_bool zero_is_max;              /* RW; Range:[0, 1]; Format:1.0; Step 0 corresponds to max aperture or not,
                                         it's related to the specific iris */
    hi_u16  total_step;               /* RW; Range:[1, 1024]; Format:16.0; Total steps of  Piris's aperture,
                                         it's related to the specific iris */
    hi_u16  step_count;               /* RW; Range:[1, 1024]; Format:16.0; Used steps of Piris's aperture.
                                         when Piris's aperture is too small, the F number precision is not enough,
                                         so not all the steps can be used. It's related to the specific iris */
    hi_u16 step_fno_table[HI_ISP_AI_MAX_STEP_FNO_NUM]; /* RW;
                                                    Range:[1, 1024]; Format:16.0; Step-F number mapping table.
                                                    F1.0 is expressed as 1024, F32 is expressed as 1,
                                                    it's related to the specific iris */
    hi_isp_iris_f_no  max_iris_fno_target;  /* RW; Range:[0, 10]; Format:4.0;
                                               Max F number of Piris's aperture, it's related to the specific iris */
    hi_isp_iris_f_no  min_iris_fno_target;  /* RW; Range:[0, 10]; Format:4.0;
                                               Min F number of Piris's aperture, it's related to the specific iris */

    hi_bool fno_ex_valid;           /* RW; Range:[0, 1]; Format:1.0; use equivalent gain to present FNO or not */
    hi_u32  max_iris_fno_target_linear;  /* RW; Range:[1, 1024]; Format:16.0;
                                            Max equivalent gain of F number of Piris's aperture,
                                            only used when bFNOExValid is true, it's related to the specific iris */
    hi_u32  min_iris_fno_target_linear;  /* RW; Range:[1, 1024]; Format:16.0;
                                            Min equivalent gain of F number of Piris's aperture,
                                            only used when bFNOExValid is true, it's related to the specific iris */
} hi_isp_piris_attr;

/* Defines the MI attribute */
typedef struct {
    hi_u32  hold_value;           /* RW; Range:[0, 1000]; Format:32.0; iris hold value for DC-iris */
    hi_isp_iris_f_no  iris_fno;   /* RW; Range:[0, 10]; Format:4.0; the F number of P-iris */
} hi_isp_mi_attr;

/*
  Defines the ISP iris status
  0 = In this mode can set the MI holdvalue
  1 = Open Iris to the max
  2 = Close Iris to the min
 */
typedef enum {
    HI_ISP_IRIS_KEEP  = 0,
    HI_ISP_IRIS_OPEN  = 1,
    HI_ISP_IRIS_CLOSE = 2,
    HI_ISP_IRIS_BUTT
} hi_isp_iris_status;

/* Defines the iris attribute */
typedef struct {
    hi_bool enable;                    /* RW;iris enable/disable */
    hi_op_mode     op_type;        /* auto iris or manual iris */
    hi_isp_iris_type   iris_type;      /* DC-iris or P-iris */
    hi_isp_iris_status iris_status;    /* RW; status of Iris */
    hi_isp_mi_attr     mi_attr;
} hi_isp_iris_attr;

/* AE structs */
/*
  Defines the AE mode
  0 = Automatic frame rate reduction mode (slow shutter mode)
  1 = Constant frame rate mode
 */
typedef enum {
    HI_ISP_AE_MODE_SLOW_SHUTTER = 0,
    HI_ISP_AE_MODE_FIX_FRAME_RATE  = 1,
    HI_ISP_AE_MODE_BUTT
} hi_isp_ae_mode;

/*
  Defines the AE exposure policy mode
  0 = Highlight first exposure mode
  1 = Lowlight first exposure mode
 */
typedef enum {
    HI_ISP_AE_EXP_HIGHLIGHT_PRIOR = 0,
    HI_ISP_AE_EXP_LOWLIGHT_PRIOR  = 1,
    HI_ISP_AE_STRATEGY_MODE_BUTT
} hi_isp_ae_strategy;

/* Defines the maximum exposure time or gain and minimum exposure time or gain */
typedef struct {
    hi_u32 max;  /* RW;Range:[0,0xFFFFFFFF];Format:32.0;Maximum value */
    hi_u32 min;  /* RW;Range:[0,0xFFFFFFFF];Format:32.0;limited Range:[0,u32Max],Minimum value */
} hi_isp_ae_range;

/* Defines the ISP exposure delay attribute */
typedef struct {
    hi_u16 black_delay_frame;    /* RW; Range:[0, 65535]; Format:16.0; AE black delay frame count */
    hi_u16 white_delay_frame;    /* RW; Range:[0, 65535]; Format:16.0; AE white delay frame count */
} hi_isp_ae_delay;

/*
  Defines the anti-flicker mode.
  0 = The exposure time is fixed to be the multiplies of 1/(2*frequency) sec,
    it may lead to over exposure in the high-luminance environments.
  1 = The anti flicker may be closed to avoid over exposure in the high-luminance environments.
 */
typedef enum {
    HI_ISP_ANTIFLICKER_NORMAL_MODE = 0x0,
    HI_ISP_ANTIFLICKER_AUTO_MODE   = 0x1,
    HI_ISP_ANTIFLICKER_MODE_BUTT
} hi_isp_antiflicker_mode;

/* Defines the anti-flicker attribute */
typedef struct {
    hi_bool enable;      /* RW; Range:[0, 1]; Format:1.0; */
    hi_u8   frequency;   /* RW; Range:[0, 255]; Format:8.0;
                            usually this value is 50 or 60  which is the frequency of the AC power supply */
    hi_isp_antiflicker_mode  mode;
} hi_isp_antiflicker;

typedef struct {
    hi_bool enable; /* RW; Range:[0, 1]; Format:1.0; */

    hi_u8   luma_diff; /* RW; Range:[0, 100]; Format:8.0; if subflicker mode enable, current luma is less than AE
                          compensation plus LumaDiff, AE will keep min antiflicker shutter time
                          (for example: 1/100s or 1/120s) to avoid flicker. while current luma is larger than AE
                          compensation plus the LumaDiff, AE will reduce shutter time to avoid over-exposure and
                          introduce flicker in the picture */
} hi_isp_subflicker;

/*
 * Defines the ISP FSWDR operating mode
 * 0 = Normal FSWDR mode
 * 1 = Long frame mode, only effective in LINE_WDR,
 * when running in this mode FSWDR module only output the long frame data
 */
typedef enum {
    HI_ISP_FSWDR_NORMAL_MODE          = 0x0,
    HI_ISP_FSWDR_LONG_FRAME_MODE      = 0x1,
    HI_ISP_FSWDR_AUTO_LONG_FRAME_MODE = 0x2, /* Auto long frame mode, only effective in LINE_WDR,  When running in this
                                             mode, normal WDR and long frame mode would auto switch */
    HI_ISP_FSWDR_MODE_BUTT
} hi_isp_fswdr_mode;

typedef struct {
    /* base parameter */
    hi_isp_ae_range exp_time_range;   /* RW; Range:[0x0, 0xFFFFFFFF]; Format:32.0; sensor exposure time (unit: us),
                                         it's related to the specific sensor */
    hi_isp_ae_range a_gain_range;     /* RW; Range:[0x400, 0xFFFFFFFF]; Format:22.10;
                                         sensor analog gain (unit: times, 10bit precision), it's related to the
                                         specific sensor */
    hi_isp_ae_range d_gain_range;     /* RW; Range:[0x400, 0xFFFFFFFF]; Format:22.10;
                                         sensor digital gain (unit: times, 10bit precision), it's related to the
                                         specific sensor */
    hi_isp_ae_range ispd_gain_range;  /* RW; Range:[0x400, 0x40000]; Format:22.10;
                                         ISP digital gain (unit: times, 10bit precision), it's related to the
                                         ISP digital gain range */
    hi_isp_ae_range sys_gain_range;   /* RW; Range:[0x400, 0xFFFFFFFF]; Format:22.10;
                                         system gain (unit: times, 10bit precision), it's related to the specific sensor
                                         and ISP Dgain range */
    hi_u32 gain_threshold;            /* RW; Range:[0x400, 0xFFFFFFFF]; Format:22.10;
                                         Gain threshold for slow shutter mode (unit: times, 10bit precision) */

    hi_u8   speed;                   /* RW; Range:[0x0, 0xFF]; Format:8.0;
                                        AE adjust step for dark scene to bright scene switch */
    hi_u16  black_speed_bias;        /* RW; Range:[0x0, 0xFFFF]; Format:16.0;
                                        AE adjust step bias for bright scene to dark scene switch */
    hi_u8   tolerance;               /* RW; Range:[0x0, 0xFF]; Format:8.0; AE adjust tolerance */
    hi_u8   compensation;            /* RW; Range:[0x0, 0xFF]; Format:8.0; AE compensation */
    hi_u16  ev_bias;                 /* RW; Range:[0x0, 0xFFFF]; Format:16.0; AE EV bias */
    hi_isp_ae_strategy ae_strategy_mode;  /* RW; Range:[0, 1]; Format:1.0; Support Highlight prior or Lowlight prior */
    hi_u16  hist_ratio_slope;       /* RW; Range:[0x0, 0xFFFF]; Format:16.0; AE hist ratio slope */
    hi_u8   max_hist_offset;        /* RW; Range:[0x0, 0xFF]; Format:8.0; Max hist offset */

    hi_isp_ae_mode     ae_mode;     /* RW; Range:[0, 1]; Format:1.0; AE mode(slow shutter/fix framerate)(onvif) */
    hi_isp_antiflicker antiflicker;
    hi_isp_subflicker  subflicker;
    hi_isp_ae_delay    ae_delay_attr;

    hi_bool manual_exp_value;     /* RW; Range:[0, 1]; Format:1.0; manual exposure value or not */
    hi_u32  exp_value;            /* RW; Range:(0x0, 0xFFFFFFFF]; Format:32.0; manual exposure value */

    hi_isp_fswdr_mode fswdr_mode; /* RW; Range:[0, 2]; Format:2.0; FSWDR running mode */
    hi_bool wdr_quick;            /* RW; Range:[0, 1]; Format:1.0; WDR use delay strategy or not;
                                     If is true, WDR AE adjust will be faster */

    hi_u16  iso_cal_coef;     /* RW; Range:[0x0, 0xFFFF]; Format:8.8;
                                 The coefficient between Standard ISO and origin ISO (unit: times, 8bit precision) */
} hi_isp_ae_attr;

typedef struct {
    hi_op_mode exp_time_op_type;
    hi_op_mode a_gain_op_type;
    hi_op_mode d_gain_op_type;
    hi_op_mode ispd_gain_op_type;

    hi_u32 exp_time;   /* RW; Range:[0x0, 0xFFFFFFFF]; Format:32.0;
                          sensor exposure time (unit: us), it's related to the specific sensor */
    hi_u32 a_gain;     /* RW; Range:[0x400, 0xFFFFFFFF]; Format:22.10;
                          sensor analog gain (unit: times, 10bit precision), it's related to the specific sensor */
    hi_u32 d_gain;     /* RW; Range:[0x400, 0xFFFFFFFF]; Format:22.10;
                          sensor digital gain (unit: times, 10bit precision), it's related to the specific sensor */
    hi_u32 isp_d_gain; /* RW; Range:[0x400, 0x40000]; Format:22.10;
                          ISP digital gain(unit: times, 10bit precision), it's related to the ISP digital gain range */
} hi_isp_me_attr;

typedef struct {
    hi_bool         bypass;      /* RW; Range:[0, 1]; Format:1.0; */
    hi_op_mode  op_type;
    hi_u8     ae_run_interval;   /* RW; Range:[0x1, 0xFF]; Format:8.0; set the AE run interval */
    hi_bool   hist_stat_adjust;  /* RW; Range:[0, 1]; Format:1.0;
                                    HI_TRUE: 256 bin histogram statistic config will adjust when large red or blue
                                    area detected.
                                    HI_FALSE: 256 bin histogram statistic config will not change */
    hi_bool   ae_route_ex_valid;   /* RW; Range:[0, 1]; Format:1.0; use extend AE route or not */
    hi_isp_me_attr   manual_attr;
    hi_isp_ae_attr   auto_attr;
    hi_isp_prior_frame prior_frame; /* RW; Range:[0, 3]; Format:1.0; AE prior frame */
    hi_bool ae_gain_sep_cfg; /* RW; Range:[0, 1]; Format:1.0; long and short frame gain separately configure or not */
    hi_bool advance_ae; /* RW; Range:[0, 1]; Format:1.0;  open advance AE or not */
} hi_isp_exposure_attr;

#define HI_ISP_AE_ROUTE_MAX_NODES                    16
typedef struct {
    hi_u32  int_time;         /* RW; Range:(0x0, 0xFFFFFFFF]; Format:32.0;
                                 sensor exposure time (unit: us), it's related to the specific sensor */
    hi_u32  sys_gain;         /* RW; Range:[0x400, 0xFFFFFFFF]; Format:22.10;
                                 system gain (unit: times, 10bit precision), it's related to the specific sensor and
                                 ISP Dgain range */
    hi_isp_iris_f_no iris_fno;  /* RW; Range:[0, 10]; Format:4.0;
                                   the F number of the iris's aperture, only support for Piris */
    hi_u32  iris_fno_lin;       /* RW; Range:[0x1, 0x400]; Format:32.0;
                                   the equivalent gain of F number of the iris's aperture, only support for Piris */
} hi_isp_ae_route_node;

typedef struct {
    hi_u32 total_num;     /* RW; Range:[0, 0x10]; Format:8.0; total node number of AE route */
    hi_isp_ae_route_node route_node[HI_ISP_AE_ROUTE_MAX_NODES];
} hi_isp_ae_route;

#define HI_ISP_AE_ROUTE_EX_MAX_NODES                 16
typedef struct {
    hi_u32  int_time;     /* RW; Range:(0x0, 0xFFFFFFFF]; Format:32.0;
                             sensor exposure time (unit: us), it's related to the specific sensor */
    hi_u32  a_gain;       /* RW; Range:[0x400, 0x3FFFFF]; Format:22.10;
                             sensor analog gain (unit: times, 10bit precision), it's related to the specific sensor */
    hi_u32  d_gain;       /* RW; Range:[0x400, 0x3FFFFF]; Format:22.10;
                             sensor digital gain (unit: times, 10bit precision), it's related to the specific sensor */
    hi_u32  isp_d_gain;   /* RW; Range:[0x400, 0x40000]; Format:22.10;
                             ISP digital gain (unit: times, 10bit precision) */
    hi_isp_iris_f_no  iris_fno;  /* RW; Range:[0, 10]; Format:4.0;
                                    the F number of the iris's aperture, only support for Piris */
    hi_u32  iris_fno_lin;  /* RW; Range:[0x1, 0x400]; Format:32.0;
                              the equivalent gain of F number of the iris's aperture, only support for Piris */
} hi_isp_ae_route_ex_node;

typedef struct {
    hi_u32 total_num;       /* RW; Range:[0, 0x10]; Format:8.0; total node number of extend AE route */
    hi_isp_ae_route_ex_node route_ex_node[HI_ISP_AE_ROUTE_EX_MAX_NODES];
} hi_isp_ae_route_ex;

typedef struct {
    hi_u32  exp_time;                   /* R; Range:[0x0, 0xFFFFFFFF]; Format:32.0; */
    hi_u32  short_exp_time;             /* R; Range:[0x0, 0xFFFFFFFF]; Format:32.0; */
    hi_u32  median_exp_time;            /* R; Range:[0x0, 0xFFFFFFFF]; Format:32.0; */
    hi_u32  long_exp_time;              /* R; Range:[0x0, 0xFFFFFFFF]; Format:32.0; */
    hi_u32  a_gain;                     /* R; Range:[0x400, 0xFFFFFFFF]; Format:22.10; */
    hi_u32  d_gain;                     /* R; Range:[0x400, 0xFFFFFFFF]; Format:22.10; */
    hi_u32  a_gain_sf;                  /* R; Range:[0x400, 0xFFFFFFFF]; Format:22.10; */
    hi_u32  d_gain_sf;                  /* R; Range:[0x400, 0xFFFFFFFF]; Format:22.10; */
    hi_u32  isp_d_gain;                 /* R; Range:[0x400, 0xFFFFFFFF]; Format:22.10; */
    hi_u32  exposure;                   /* R; Range:[0x40, 0xFFFFFFFF]; Format:26.6; */
    hi_bool exposure_is_max;            /* R; Range:[0, 1]; Format:1.0; */
    hi_s16  hist_error;                 /* R; Range:[-32768, 32767]; Format:16.0; */
    hi_u32  ae_hist1024_value[HI_ISP_HIST_NUM]; /* R; Range:[0x0, 0xFFFFFFFF]; Format:32.0;
                                            1024 bins histogram for channel 1 */

    hi_u8  ave_lum;                 /* R; Range:[0x0, 0xFF]; Format:8.0; */
    hi_u32 lines_per500ms;          /* R; Range:[0x0, 0xFFFFFFFF]; Format:32.0; exposure lines per 500ms */
    hi_u32 piris_fno;               /* R; Range:[0x0, 0x400]; Format:32.0; */
    hi_u32 fps;                     /* R; Range:[0x0, 0xFFFFFFFF]; Format:32.0; actual fps */
    hi_u32 iso;                     /* R; Range:[0x64, 0xFFFFFFFF]; Format:32.0; */
    hi_u32 isosf;                   /* R; Range:[0x64, 0xFFFFFFFF]; Format:32.0; */
    hi_u32 iso_calibrate;           /* R; Range:[0x0, 0xFFFFFFFF]; Format:32.0; */
    hi_u32 ref_exp_ratio;           /* R; Range:[0x40, 0x4000]; Format:26.6; */
    hi_u16 wdr_exp_coef;            /* R; Range:[0x0, 0x400]; Format:6.10; 0x400 means 1 times */
    hi_u32 first_stable_time;       /* R; Range:[0x0, 0xFFFFFFFF]; Format:32.0; AE first stable time for quick start */
    hi_u32 quick_star_iso;          /* R; Range:[0x64, 0xFFFFFFFF]; Format:32.0; */
    hi_isp_ae_route ae_route;       /* R; Actual AE route */
    hi_isp_ae_route_ex ae_route_ex; /* R; Actual AE route_ex */
    hi_isp_ae_route ae_route_sf;    /* R; Actual AE route_sf */
    hi_isp_ae_route_ex ae_route_sf_ex; /* R; Actual AE route_sf_ex */
} hi_isp_exp_info;

typedef struct {
    hi_u32 reg_addr;
    hi_u32 reg_value;
} hi_isp_exp_param_reg;

typedef struct {
    hi_u32 tar_fps;
    hi_u32 tar_isp_dgain; /* 10 bit */
    hi_isp_exp_param_reg time_reg[10]; /* default 10 */
    hi_isp_exp_param_reg again_reg[10]; /* default 10 */
    hi_isp_exp_param_reg dgain_reg[10]; /*  default 10 */
} hi_isp_exp_conv_param;

#define HI_ISP_EXP_RATIO_NUM                             3
typedef struct {
    hi_op_mode exp_ratio_type;   /* RW; Range:[0, 1]; Format:1.0;
                                        HI_OP_MODE_AUTO: The ExpRatio used in ISP is generated by firmware;
                                        HI_OP_MODE_MANUAL: The ExpRatio used in ISP is set by u32ExpRatio */
    hi_u32 exp_ratio[HI_ISP_EXP_RATIO_NUM]; /* RW; Range:[0x40, 0xFFF]; Format:26.6; 0x40 means 1 times.
                                        When enExpRatioType is HI_OP_MODE_AUTO, u32ExpRatio is invalid.
                                        When enExpRatioType is HI_OP_MODE_MANUAL, u32ExpRatio is quotient of
                                        long exposure time / short exposure time. */
    hi_u32 exp_ratio_max;           /* RW; Range:[0x40, 0x4000]; Format:26.6; 0x40 means 1 times.
                                       When enExpRatioType is HI_OP_MODE_AUTO, u32ExpRatioMax is max(upper limit) of
                                       ExpRatio generated by firmware.
                                       When enExpRatioType is HI_OP_MODE_MANUAL, u32ExpRatioMax is invalid. */
    hi_u32 exp_ratio_min;          /* RW; Range:[0x40, 0x4000]; Format:26.6; limited Range:[0x40, u32ExpRatioMax],
                                      0x40 means 1 times.
                                      When enExpRatioType is HI_OP_MODE_AUTO, u32ExpRatioMin is min(lower limit) of
                                      ExpRatio generated by firmware.
                                      When enExpRatioType is HI_OP_MODE_MANUAL, u32ExpRatioMin is invalid. */
    hi_u16 tolerance;              /* RW; Range:[0x0, 0xFF]; Format:8.0;et the dynamic range tolerance.
                                      Format: unsigned 6.2-bit fixed-point. 0x4 means 1dB. */
    hi_u16 speed;                  /* RW; Range:[0x0, 0xFF]; Format:8.0; exposure ratio adjust speed */
    hi_u16 ratio_bias;             /* RW; Range:[0x0, 0xFFFF]; Format:16.0; exposure ratio bias */
    hi_u16 high_light_target;      /* RW; Range:[0x0, 0x400]; Format:10.0; the high light target. */
    hi_u16 exp_coef_min;       /* RW; Range:[0x0, 0x400]; Format:6.10; 0x400 means 1 times; min exposure coefficient */
} hi_isp_wdr_exposure_attr;

typedef struct {
    hi_op_mode exp_hdr_lv_type;   /* RW; Range:[0, 1]; Format:1.0;
                                         HI_OP_MODE_AUTO: The ExpHDRLv used in ISP is generated by firmware;
                                         HI_OP_MODE_MANUAL: The ExpHDRLv used in ISP is set by u32ExpHDRLv */
    hi_u32 exp_hdr_lv;             /* RW; Range:[0x40, 0x400]; Format:26.6; 0x40 means 1 times.
                                      When enExpHDRLvType is HI_OP_MODE_AUTO, u32ExpHDRLv is invalid.
                                      When enExpHDRLvType is HI_OP_MODE_MANUAL, u32ExpHDRLv is the quotient of
                                      exposure / linear exposure. */
    hi_u32 exp_hdr_lv_max;         /* RW; Range:[0x40, 0x400]; Format:26.6; 0x40 means 1 times.
                                      When enExpHDRLvType is HI_OP_MODE_AUTO, u32ExpHDRLvMax is max(upper limit) of
                                      ExpHDRLv generated by firmware.
                                      When enExpHDRLvType is HI_OP_MODE_MANUAL, u32ExpHDRLvMax is invalid. */
    hi_u32 exp_hdr_lv_min;         /* RW; Range:[0x40, 0x400]; Format:26.6; limited range:[0x40, u32ExpHDRLvMax],
                                      0x40 means 1 times.
                                      When enExpHDRLvType is HI_OP_MODE_AUTO, u32ExpHDRLvMin is min(lower limit) of
                                      ExpHDRLv generated by firmware.
                                      When enExpHDRLvType is HI_OP_MODE_MANUAL, u32ExpHDRLvMin is invalid. */
    hi_u32 exp_hdr_lv_weight;      /* RW; Range:[0x0, 0x400]; Format:16.0; exposure HDR level weight. */
} hi_isp_hdr_exposure_attr;

typedef struct {
    hi_bool enable;               /* RW; Range:[0, 1]; Format:1.0; smart ae enable or not */
    hi_bool ir_mode;              /* RW; Range:[0, 1]; Format:1.0; smart ae IR mode or not */
    hi_op_mode smart_exp_type;  /* RW; Range:[0, 1]; Format:1.0;
                                       HI_OP_MODE_AUTO: The ExpCoef used in ISP is  generated by firmware;
                                       HI_OP_MODE_MANUAL: The ExpCoef used in ISP is set by u32ExpCoef */
    hi_u16  exp_coef;             /* RW; Range:[0x0, 0xFFFF]; Format:6.10; 0x400 means 1 times.
                                     When enExpHDRLvType is HI_OP_MODE_AUTO, u32ExpCoef is invalid.
                                     When enExpHDRLvType is HI_OP_MODE_MANUAL,
                                     u32ExpCoef is the quotient of exposure. */
    hi_u8   luma_target;          /* RW; Range:[0x0, 0xFF]; Format:8.0; luma target of smart ae. */
    hi_u16  exp_coef_max;         /* RW; Range:[0x0, 0xFFFF]; Format:6.10; 0x400 means 1 times.
                                     When enExpHDRLvType is HI_OP_MODE_AUTO,u32ExpCoefvMax is max(upper limit)of ExpCoef
                                     generated by firmware.
                                     When enExpHDRLvType is HI_OP_MODE_MANUAL, u32ExpCoefMax is invalid. */
    hi_u16  exp_coef_min;         /* RW; Range:[0x0, 0xFFFF]; Format:6.10; 0x400 means 1 times.
                                     When enExpHDRLvType is HI_OP_MODE_AUTO,u32ExpCoefMax is min(lower limit) of ExpCoef
                                     generated by firmware.
                                     When enExpHDRLvType is HI_OP_MODE_MANUAL, u32ExpCoefMin is invalid. */
    hi_u8   smart_interval;       /* RW; Range:[0x1, 0xFF]; Format:8.0; smart ae run interval */
    hi_u8   smart_speed;          /* RW; Range:[0x0, 0xFF]; Format:8.0; smart ae adjust step */
    hi_u16  smart_delay_num;      /* RW; Range:[0x0, 0x400]; Format:16.0; smart ae adjust delay frame num */
} hi_isp_smart_exposure_attr;

/*
  Defines the AWB algorithm type
  0 = Improved gray world algorithm.
  1 = AWB algorithm that classifies the statistics and re-filters the white blocks
 */
typedef enum {
    HI_ISP_AWB_ALG_LOWCOST = 0,
    HI_ISP_AWB_ALG_ADVANCE = 1,
    HI_ISP_AWB_ALG_BUTT
} hi_isp_awb_alg_type;

/*
  Defines the AWB policy in the multi-illuminant scenario
  0 = Automatic saturation adjustment in the multi-illuminant scenario
  1 = Automatic ccm adjustment in the multi-illuminant scenario
 */
typedef enum {
    HI_ISP_AWB_MULTI_LS_SAT = 0,
    HI_ISP_AWB_MULTI_LS_CCM = 1,
    HI_ISP_AWB_MULTI_LS_BUTT
} hi_isp_awb_multi_ls_type;

typedef enum {
    HI_ISP_AWB_INDOOR_MODE = 0,
    HI_ISP_AWB_OUTDOOR_MODE = 1,
    HI_ISP_AWB_INDOOR_OUTDOOR_BUTT
} hi_isp_awb_indoor_outdoor_status;

/* Defines the AWB gain range */
typedef struct {
    hi_bool enable;
    hi_op_mode   op_type;

    hi_u16 high_rg_limit; /* RW; Range:[0x0, 0xFFF]; Format:4.8;
                             in Manual Mode, user define the Max Rgain of High Color Temperature */
    hi_u16 high_bg_limit; /* RW; Range:[0x0, 0xFFF]; Format:4.8;
                             in Manual Mode, user define the Min Bgain of High Color Temperature */
    hi_u16 low_rg_limit;  /* RW; Range:[0x0, 0xFFF]; Format:4.8; limited range:[0x0, u16HighRgLimit),
                             in Manual Mode, user define the Min Rgain of Low Color Temperature */
    hi_u16 low_bg_limit;  /* RW; Range:[0, 0xFFF]; Format:4.8; limited Range:(u16HighBgLimit, 0xFFF],
                             in Manual Mode, user define the Max Bgain of Low Color Temperature */
} hi_isp_awb_ct_limit_attr;

typedef struct {
    hi_bool enable;                 /* RW; Range:[0, 1]; Format:1.0; Outdoor/Indoor scenario determination enable */
    hi_op_mode   op_type;
    hi_isp_awb_indoor_outdoor_status outdoor_status; /* RW;User should config indoor or outdoor status in Manual Mode */
    hi_u32 out_thresh;         /* RW;Format:32.0;shutter time(in us) to judge indoor or outdoor */
    hi_u16 low_start;          /* RW;Format:16.0;5000K is recommend */
    hi_u16 low_stop;           /* RW;Format:16.0;limited range:(0, low_start),
                                  4500K is recommend, should be smaller than low_start */
    hi_u16 high_start;         /* RW;Format:16.0;limited range:(low_start, 0xFFFF],
                                  6500K is recommend, shoule be larger than low_start */
    hi_u16 high_stop;          /* RW;Format:16.0;limited range:(high_start, 0xFFFF],
                                  8000K is recommend, should be larger than high_start */
    hi_bool green_enhance_en;  /* RW; Range:[0, 1]; Format:1.0;If this is enabled, Green channel will be enhanced
                                  based on the ratio of green plant */
    hi_u8   out_shift_limit;   /* RW; Range:[0, 0xFF]; Format:8.0;Max white point zone distance to Planckian Curve */
} hi_isp_awb_in_out_attr;

typedef struct {
    hi_bool enable;                         /* RW; Range:[0, 1]; Format:1.0;
                                               If enabled, statistic parameter cr, cb will change according to iso */

    hi_u16  cr_max[HI_ISP_AUTO_ISO_NUM]; /* RW; Range:[0x0, 0xFFF]; au16CrMax[i] >= au16CrMin[i] */
    hi_u16  cr_min[HI_ISP_AUTO_ISO_NUM]; /* RW; Range:[0x0, 0xFFF] */
    hi_u16  cb_max[HI_ISP_AUTO_ISO_NUM]; /* RW; Range:[0x0, 0xFFF]; au16CbMax[i] >= au16CbMin[i] */
    hi_u16  cb_min[HI_ISP_AUTO_ISO_NUM]; /* RW; Range:[0x0, 0xFFF] */
} hi_isp_awb_cbcr_track_attr;

#define HI_ISP_AWB_LUM_HI_ISP_HIST_NUM                          6
/* Defines the parameters of the luminance histogram statistics for white balance */
typedef struct {
    hi_bool enable;                        /* RW; Range:[0, 1]; Format:1.0;
                                              If enabled, zone weight to awb is combined with zone luma */
    hi_op_mode   op_type;              /* In auto mode, the weight distribution follows Gaussian distribution */
    hi_u8   hist_thresh[HI_ISP_AWB_LUM_HI_ISP_HIST_NUM]; /* RW; Range:[0x0, 0xFF]; Format:8.0;
                                              In manual mode, user define luma thresh, thresh[0] is 0,
                                              thresh[5] is 0xFF,  thresh[i] greater or equal to  thresh[i-1] */
    hi_u16  hist_wt[HI_ISP_AWB_LUM_HI_ISP_HIST_NUM];     /* RW; Range:[0x0, 0xFFFF]; Format:16.0;
                                              user can define luma weight in both manual and auto mode. */
} hi_isp_awb_lum_histgram_attr;

/* Defines the information about a separate illuminant */
typedef struct {
    hi_u16 white_r_gain;  /* RW;Range:[0x0, 0xFFF]; Format:4.8; G/R of White points at this light source */
    hi_u16 white_b_gain;  /* RW;Range:[0x0, 0xFFF]; Format:4.8; G/B of White points at this light source */
    hi_u16 exp_quant;     /* RW;shutter time * again * dgain >> 4, Not support Now */
    hi_u8  light_status;  /* RW;Range:[0,2]; Format:2.0; idle  1:add light source  2:delete sensitive color */
    hi_u8  radius;        /* RW;Range:[0x0, 0xFF];  Format:8.0; Radius of light source, */
} hi_isp_awb_extra_light_source_info;

#define HI_ISP_AWB_LS_NUM                                4
#define HI_ISP_AWB_MULTI_CT_NUM                          8
/* extended AWB attributes */
typedef struct {
    hi_u8  tolerance;        /* RW; Range:[0x0, 0xFF]; Format:8.0; AWB adjust tolerance */
    hi_u8  zone_radius;      /* RW; Range:[0x0, 0xFF]; Format:8.0; radius of AWB blocks */
    hi_u16 curve_l_limit;    /* RW; Range:[0x0, 0x100]; Format:9.0; Left limit of AWB Curve */
    hi_u16 curve_r_limit;    /* RW; Range:[0x100, 0xFFF]; Format:12.0; Right Limit of AWB Curve */

    hi_bool  extra_light_en; /* RW; Range:[0, 1]; Format:1.0; Enable special light source function */
    hi_isp_awb_extra_light_source_info  light_info[HI_ISP_AWB_LS_NUM];
    hi_isp_awb_in_out_attr              in_or_out;

    hi_bool multi_light_source_en;           /* RW; Range:[0, 1]; Format:1.0; If enabled, awb will do special process
                                                in multi light source environment */
    hi_isp_awb_multi_ls_type multi_ls_type;  /* Saturation or CCM Tuning */
    hi_u16  multi_ls_scaler;                 /* RW; Range:[0x0, 0x100]; Format:12.0;
                                                In saturation type, it means the max saturation it can achieve,
                                                in ccm type, it means the strength of multi process. */
    hi_u16  multi_ct_bin[HI_ISP_AWB_MULTI_CT_NUM];  /* RW; Range:[0, 0XFFFF]; Format:16.0;
                                                AWB Support divide the color temperature range by 8 bins */
    hi_u16  multi_ct_wt[HI_ISP_AWB_MULTI_CT_NUM];   /* RW; Range:[0x0, 0x400];Weight for different color temperature,
                                                same value of 8 means CT weight does't work, */

    hi_bool fine_tun_en;       /* RW; Range:[0x0, 0x1]; Format:1.0;If enabled, skin color scene will be optimized */
    hi_u8   fine_tun_strength; /* RW; Range:[0x0, 0xFF]; Format:8.0;
                                  larger value means better performance of skin color scene,
                                  but will increase error probability in low color temperature scene */
} hi_isp_awb_attr_ex;

#define HI_ISP_AWB_CURVE_PARA_NUM                        6
typedef struct {
    hi_bool enable;               /* RW; Range:[0x0, 0x1]; Format:1.0;If AWB is disabled, static wb gain will be used,
                                     otherwise auto wb gain will be used */

    hi_u16 ref_color_temp;        /* RW; Range:[0x0, 0xFFFF]; Format:16.0; Calibration Information */
    hi_u16 static_wb[HI_ISP_BAYER_CHN_NUM];   /* RW; Range:[0x0, 0xFFF]; Format:12.0;Calibration Information */
    hi_s32 curve_para[HI_ISP_AWB_CURVE_PARA_NUM]; /* RW; Format:32.0;Calibration Information,
                                              limited Range:as32CurvePara[3] != 0, as32CurvePara[4]==128 */

    hi_isp_awb_alg_type       alg_type;

    hi_u8  rg_strength;     /* RW; Range: [0x0, 0xFF]; Format:8.0; AWB Strength of R Channel */
    hi_u8  bg_strength;     /* RW; Range: [0x0, 0xFF]; Format:8.0; AWB Strength of B Channel */
    hi_u16 speed;           /* RW; Range: [0x0, 0xFFF]; Format:12.0; Convergence speed of AWB  */
    hi_u16 zone_sel;        /* RW; Range: [0, 255]; Format:8.0; A value of 0 or 0xFF means global AWB,
                               A value between 0 and 0xFF means zoned AWB */
    hi_u16 high_color_temp; /* RW; Range: [0, 65535]; Format:16.0; AWB max temperature, Recommended: [8500, 10000] */
    hi_u16 low_color_temp;  /* RW; Range: [0, 65535]; Format:16.0; Limited Range:[0, u16HighColorTemp),
                               AWB min temperature, Recommended: [2000, 2500] */
    hi_isp_awb_ct_limit_attr ct_limit;
    hi_bool shift_limit_en;  /* RW; Range: [0, 1]; Format:1.0; If enabled, when the statistic information is out of
                                range, it should be project back */
    hi_u8  shift_limit;      /* RW; Range: [0x0, 0xFF];  Format:8.0; planckian curve range, Recommended: [0x30, 0x50] */
    hi_bool gain_norm_en;    /* RW; Range: [0, 1]; Format:1.0; if enabled, the min of RGB gain is fixed. */
    hi_bool natural_cast_en; /* RW, Range: [0, 1]; Format:1.0; if enabled, the color performance will be natural in
                                lowlight and low color temperature */

    hi_isp_awb_cbcr_track_attr cb_cr_track;
    hi_isp_awb_lum_histgram_attr luma_hist;
    hi_bool awb_zone_wt_en;      /* RW, Range: [0, 1]; Format:1.0; if enabled, user can set weight for each zones */
    /* RW; Range: [0, 255]; Format:8.0;Zone Wt Table */
    hi_u8   zone_wt[(HI_ISP_AWB_ZONE_ORIG_ROW) * (HI_ISP_AWB_ZONE_ORIG_COLUMN)];
} hi_isp_awb_attr;

typedef struct {
    hi_u16 r_gain;            /* RW; Range: [0x0, 0xFFF]; Format:4.8; Multiplier for R  color channel  */
    hi_u16 gr_gain;           /* RW; Range: [0x0, 0xFFF]; Format:4.8; Multiplier for Gr color channel */
    hi_u16 gb_gain;           /* RW; Range: [0x0, 0xFFF]; Format:4.8; Multiplier for Gb color channel */
    hi_u16 b_gain;            /* RW; Range: [0x0, 0xFFF]; Format:4.8; Multiplier for B  color channel */
} hi_isp_mwb_attr;

typedef enum {
    HI_ISP_ALG_AWB         = 0,
    HI_ISP_ALG_AWB_SPEC    = 1,
    HI_ISP_ALG_BUTT
} hi_isp_awb_alg;

typedef struct {
    hi_bool bypass;              /* RW; Range: [0, 1];  Format:1.0; If enabled, awb will be bypassed */
    hi_u8   awb_run_interval;    /* RW; Range: [0x1, 0xFF];  Format:8.0; set the AWB run interval */
    hi_op_mode   op_type;
    hi_isp_mwb_attr  manual_attr;
    hi_isp_awb_attr  auto_attr;
    hi_isp_awb_alg   alg_type;
} hi_isp_wb_attr;

typedef struct {
    hi_bool sat_en;               /* RW; Range: [0, 1]; Format:1.0;
                                     If bSatEn=1, the active CCM = SatMatrix * ManualMatrix,
                                     else tha active CCM =  ManualMatrix */
    hi_u16  ccm[HI_ISP_CCM_MATRIX_SIZE]; /* RW; Range: [0x0, 0xFFFF]; Format:8.8; Manul CCM matrix, */
} hi_isp_color_matrix_manual;

typedef struct {
    hi_u16 color_temp;           /* RW; Range: [500, 30000]; Format:16.0; the current color temperature */
    /* RW; Range: [0x0, 0xFFFF]; Format:8.8;CCM matrixes for different color temperature */
    hi_u16 ccm[HI_ISP_CCM_MATRIX_SIZE];
} hi_isp_color_matrix_param;

typedef struct {
    hi_bool iso_act_en;    /* RW; Range: [0, 1]; Format:1.0; if enabled, CCM will bypass in low light */
    hi_bool temp_act_en;   /* RW; Range: [0, 1]; Format:1.0; if enabled, CCM will bypass when color temperature is
                              larger than 8000K or less than 2500K */
    hi_u16  ccm_tab_num;   /* RW; Range: [0x3, 0x7]; Format:16.0; The number of CCM matrix */
    hi_isp_color_matrix_param ccm_tab[HI_ISP_CCM_MATRIX_NUM];
} hi_isp_color_matrix_auto;

typedef struct {
    hi_op_mode op_type;
    hi_isp_color_matrix_manual manual_attr;
    hi_isp_color_matrix_auto   auto_attr;
} hi_isp_color_matrix_attr;

typedef struct {
    hi_u8 saturation;           /* RW; Range: [0, 0xFF]; Format:8.0; set the manual saturation of CCM */
} hi_isp_saturation_manual;

typedef struct {
    hi_u8 sat[HI_ISP_AUTO_ISO_NUM];           /* RW; Range: [0, 0xFF]; should be decreased with increasing ISO */
} hi_isp_saturation_auto;

typedef struct {
    hi_op_mode op_type;
    hi_isp_saturation_manual manual_attr;
    hi_isp_saturation_auto   auto_attr;
} hi_isp_saturation_attr;

typedef struct {
    hi_u16 red_cast_gain;        /* RW; Range: [0x100, 0x180]; Format:4.8; R channel gain after CCM */
    hi_u16 green_cast_gain;      /* RW; Range: [0x100, 0x180]; Format:4.8; G channel gain after CCM */
    hi_u16 blue_cast_gain;       /* RW; Range: [0x100, 0x180]; Format:4.8; B channel gain after CCM */
} hi_isp_color_tone_attr;

typedef struct {
    hi_u16 r_gain;               /* R; Range: [0x0, 0xFFF]; Format:8.8;AWB result of R color channel */
    hi_u16 gr_gain;              /* R; Range: [0x0, 0xFFF]; Format:8.8; AWB result of Gr color channel */
    hi_u16 gb_gain;              /* R; Range: [0x0, 0xFFF]; Format:8.8; AWB result of Gb color channel */
    hi_u16 b_gain;               /* R; Range: [0x0, 0xFFF]; Format:8.8; AWB result of B color channel */
    hi_u16 saturation;           /* R; Range: [0x0, 0xFF];Format:8.0;Current saturation */
    hi_u16 color_temp;           /* R; Range: [0x0, 0xFFFF];Format:16.0;Detect color temperature, maybe out of color
                                    temperature range */
    hi_u16 ccm[HI_ISP_CCM_MATRIX_SIZE]; /* R; Range: [0x0, 0xFFFF];Format:16.0;Current color correction matrix */

    hi_u16 ls0_ct;               /* R; Range: [0x0, 0xFFFF];Format:16.0;color temperature of primary light source */
    hi_u16 ls1_ct;               /* R; Range: [0x0, 0xFFFF];Format:16.0;color temperature of secondary light source */
    hi_u16 ls0_area;             /* R; Range: [0x0, 0xFF];Format:8.0;area of primary light source */
    hi_u16 ls1_area;             /* R; Range: [0x0, 0xFF];Format:8.0;area of secondary light source */
    hi_u8  multi_degree;         /* R; Range: [0x0, 0xFF];0 means uniform light source, larger value means multi
                                    light source */
    hi_u16 active_shift;         /* R; Range;[0x0,0xFF] */
    hi_u32 first_stable_time;    /* R, Range: [0x0, 0xFFFFFFFF];Format:32.0;AWB first stable frame number */
    hi_isp_awb_indoor_outdoor_status in_out_status;   /* R; indoor or outdoor status */
    hi_s16 bv;                                   /* R; Range;[-32768, 32767]; Bv value */
} hi_isp_wb_info;

typedef struct {
    hi_bool is_ir_mode;
    hi_u32 exp_time;
    hi_u32 a_gain;
    hi_u32 d_gain;
    hi_u32 ispd_gain;
    hi_u32 exposure;
    hi_u32 init_iso;
    hi_u32 lines_per500ms;
    hi_u32 piris_fno;
    hi_u16 wb_r_gain;
    hi_u16 wb_g_gain;
    hi_u16 wb_b_gain;
    hi_u16 sample_r_gain;
    hi_u16 sample_b_gain;
    hi_u16 init_ccm[HI_ISP_CCM_MATRIX_SIZE];
    hi_bool ae_route_ex_valid;
    hi_bool quick_start_en;
    hi_isp_ae_route ae_route;
    hi_isp_ae_route_ex ae_route_ex;
    hi_isp_ae_route ae_route_sf;
    hi_isp_ae_route_ex ae_route_sf_ex;
} hi_isp_init_attr;
/*
  DNG cfalayout type
  1 = Rectangular (or square) layout
  2 = Staggered layout A: even columns are offset down by 1/2 row
  3 = Staggered layout B: even columns are offset up by 1/2 row
  4 = Staggered layout C: even rows are offset right by 1/2 column
  5 = Staggered layout D: even rows are offset left by 1/2 column
  6 = Staggered layout E: even rows are offset up by 1/2 row, even columns are offset left by 1/2 column
  7 = Staggered layout F: even rows are offset up by 1/2 row, even columns are offset right by 1/2 column
  8 = Staggered layout G: even rows are offset down by 1/2 row, even columns are offset left by 1/2 column
  9 = Staggered layout H: even rows are offset down by 1/2 row, even columns are offset right by 1/2 column
 */
typedef enum hiDNG_CFALAYOUT_TYPE_E {
    CFALAYOUT_TYPE_RECTANGULAR = 1,
    CFALAYOUT_TYPE_A,        /* a,b,c... not support */
    CFALAYOUT_TYPE_B,
    CFALAYOUT_TYPE_C,
    CFALAYOUT_TYPE_D,
    CFALAYOUT_TYPE_E,
    CFALAYOUT_TYPE_F,
    CFALAYOUT_TYPE_G,
    CFALAYOUT_TYPE_H,
    CFALAYOUT_TYPE_BUTT
} hi_isp_dng_cfa_layout_type;

typedef struct {
    hi_s32 numerator;   /* represents the numerator of a fraction, */
    hi_s32 denominator; /* the denominator. */
} hi_isp_dng_srational;

typedef struct {
    hi_u16 blc_repeat_rows;
    hi_u16 blc_repeat_cols;
} hi_isp_dng_blc_repeat_dim;

typedef struct {
    hi_isp_dng_srational default_scale_h;
    hi_isp_dng_srational default_scale_v;
} hi_isp_dng_default_scale;

typedef struct {
    hi_u16 repeat_pattern_dim_rows;
    hi_u16 repeat_pattern_dim_cols;
} hi_isp_dng_repeat_pattern_dim;

/* Defines the structure of dng raw format. */
typedef struct {
    hi_u8 bits_per_sample;                      /* RO;Format:8.0; Indicate the bit numbers of raw data */
    hi_u8 cfa_plane_color[HI_CFACOLORPLANE];       /* RO;Format:8.0; Indicate the planer numbers of raw data;
                                                   0:red 1:green 2: blue */
    hi_isp_dng_cfa_layout_type cfa_layout;      /* RO;Range:[1,9]; Describes the spatial layout of the CFA */
    hi_isp_dng_blc_repeat_dim blc_repeat_dim;   /* Specifies repeat pattern size for the BlackLevel */
    hi_u32 white_level;                         /* RO;Format:32.0; Indicate the WhiteLevel of the raw data */
    hi_isp_dng_default_scale default_scale;     /* Specifies the default scale factors for each direction to convert
                                                   the image to square pixels */
    hi_isp_dng_repeat_pattern_dim cfa_repeat_pattern_dim; /* Specifies the pixel number of repeat color planer in
                                                             each direction */
    /* RO;Format:8.0; Indicate the bayer start order; 0:red 1:green 2: blue */
    hi_u8 cfa_pattern[HI_ISP_BAYER_CHN_NUM];
} hi_isp_dng_raw_format;

/* Defines the structure of DNG WB gain used for calculate DNG colormatrix. */
typedef struct {
    hi_u16 r_gain;            /* RW;Range: [0x0, 0xFFF]; Multiplier for R  color channel */
    hi_u16 g_gain;            /* RW;Range: [0x0, 0xFFF]; Multiplier for G  color channel */
    hi_u16 b_gain;            /* RW;Range: [0x0, 0xFFF]; Multiplier for B  color channel */
} hi_isp_dng_wb_gain;

typedef enum {
    HI_ISP_FLICKER_TYPE_NONE = 0,
    HI_ISP_FLICKER_TYPE_UNKNOW,
    HI_ISP_FLICKER_TYPE_50HZ,
    HI_ISP_FLICKER_TYPE_60HZ,
    HI_ISP_FLICKER_TYPE_BUTT,
} hi_isp_calc_flicker_type;

typedef struct  {
    hi_u32 lines_per_second;      /* The total line number of 1 second */
} hi_isp_calc_flicker_input;

typedef struct {
    hi_isp_calc_flicker_type flicker_type;  /* The calculate result of flicker type */
} hi_isp_calc_flicker_output;

typedef struct {
    hi_u32      iso_speed_ratings;   /* CCD sensitivity equivalent to ag-hr film speedrate */
    hi_u32      exposure_time;       /* exposure time (reciprocal of shutter speed). */
    hi_u32      exposure_bias_value; /* exposure bias (compensation) value of taking picture */
    hi_u8       exposure_program;    /* exposure program that the camera used when image was taken.
                                       '1' means manual control, '2' program normal, '3' aperture priority,
                                       '4' shutter priority, '5' program creative (slow program),
                                       '6' program action(high-speed program), '7' portrait mode, '8' landscape mode */
    hi_u32      f_number;            /* the actual F-number (F-stop) of lens when the image was taken */
    hi_u32      max_aperture_value;  /* maximum aperture value of lens. */
    hi_u8       exposure_mode;       /* indicates the exposure mode set when the image was shot.
                                        0 = auto exposure,1 = manual exposure, 2 = auto bracket */
    hi_u8       white_balance;       /* indicates the white balance mode set when the image was shot.
                                        0 = auto white balance ,1 = manual white balance */
} hi_isp_dcf_update_info;
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */

#endif /* HI_COMM_ISP_ADAPT_H */
