/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: Api of mipi rx
 * Author: Hisilicon multimedia software group
 * Create: 2023-01-05
 */

#ifndef HI_MIPI_RX_H
#define HI_MIPI_RX_H

#include <linux/ioctl.h>
#include "hi_media_type.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define MIPI_LANE_NUM           8
#define LVDS_LANE_NUM           8
#define SLVS_LANE_NUM           8

#define WDR_VC_NUM              4
#define SYNC_CODE_NUM           4

#define MIPI_RX_MAX_DEV_NUM     4

#define SNS_MAX_CLK_SOURCE_NUM  4
#define SNS_MAX_RST_SOURCE_NUM  2

typedef enum {
    LANE_DIVIDE_MODE_0 = 0,  /* 8lane */
    LANE_DIVIDE_MODE_1 = 1,  /* 4lane + 4lane */
    LANE_DIVIDE_MODE_2 = 2,  /* 4lane + 2lane +2lane */
    LANE_DIVIDE_MODE_3 = 3,  /* 2lane + 2lane + 2lane + 2lane */
    LANE_DIVIDE_MODE_BUTT
} lane_divide_mode_t;

typedef enum {
    WORK_MODE_LVDS = 0x0,
    WORK_MODE_MIPI = 0x1,
    WORK_MODE_CMOS = 0x2,
    WORK_MODE_BT1120 = 0x3,
    WORK_MODE_SLVS = 0x4,
    WORK_MODE_BUTT
} work_mode_t;

typedef enum {
    INPUT_MODE_MIPI = 0x0,    /* mipi */
    INPUT_MODE_SUBLVDS = 0x1, /* SUB_LVDS */
    INPUT_MODE_LVDS = 0x2,    /* LVDS */
    INPUT_MODE_HISPI = 0x3,   /* HISPI */
    INPUT_MODE_SLVS = 0x4, // SLVS_EC
    INPUT_MODE_BUTT
} input_mode_t;

typedef enum {
    MIPI_DATA_RATE_X1 = 0, /* output 1 pixel per clock */
    MIPI_DATA_RATE_X2 = 1, /* output 2 pixel per clock */
    MIPI_DATA_RATE_BUTT
} mipi_data_rate_t;

typedef struct {
    hi_s32 x;
    hi_s32 y;
    hi_u32 width;
    hi_u32 height;
} img_rect_t;

typedef struct {
    hi_u32 width;
    hi_u32 height;
} img_size_t;

typedef enum {
    DATA_TYPE_RAW_8BIT = 0,
    DATA_TYPE_RAW_10BIT = 1,
    DATA_TYPE_RAW_12BIT = 2,
    DATA_TYPE_RAW_14BIT = 3,
    DATA_TYPE_YUV420_8BIT_NORMAL = 5,
    DATA_TYPE_YUV422_8BIT = 7,
    DATA_TYPE_YUV422_PACKED = 8, /* yuv422 8bit transform user define 16bit raw */
    DATA_TYPE_BUTT
} data_type_t;

/* MIPI D_PHY WDR MODE defines */
typedef enum {
    HI_MIPI_WDR_MODE_NONE = 0x0,
    HI_MIPI_WDR_MODE_VC = 0x1,  /* Virtual Channel */
    HI_MIPI_WDR_MODE_DT = 0x2,  /* Data Type */
    HI_MIPI_WDR_MODE_DOL = 0x3, /* DOL Mode */
    HI_MIPI_WDR_MODE_BUTT
} mipi_wdr_mode_t;
/* LVDS WDR MODE defines */

typedef struct {
    data_type_t input_data_type;  /* data type: 8/10/12/14/16 bit */
    mipi_wdr_mode_t wdr_mode;     /* MIPI WDR mode */
    short lane_id[MIPI_LANE_NUM]; /* lane_id: -1 - disable */

    union {
        short data_type[WDR_VC_NUM]; /* attribute of MIPI WDR mode. AUTO:mipi_wdr_mode_t:OT_MIPI_WDR_MODE_DT; */
    };
} mipi_dev_attr_t;

typedef enum {
    HI_SLVS_WDR_MODE_NONE     = 0x0,
    HI_SLVS_WDR_MODE_2F       = 0x1,
    HI_SLVS_WDR_MODE_DOL_2F  = 0x4
} slvs_wdr_mode_t;

typedef enum {
    HI_LVDS_WDR_MODE_NONE     = 0x0,
    HI_LVDS_WDR_MODE_2F       = 0x1,
    HI_LVDS_WDR_MODE_DOL_2F  = 0x4
} lvds_wdr_mode_t;

typedef enum {
    LVDS_SYNC_MODE_SOF = 0, /* sensor SOL, EOL, SOF, EOF */
    LVDS_SYNC_MODE_SAV,     /* SAV, EAV */
    LVDS_SYNC_MODE_BUTT
} lvds_sync_mode_t;

typedef enum {
    LVDS_VSYNC_NORMAL = 0x00,
    LVDS_VSYNC_SHARE = 0x01,
    LVDS_VSYNC_HCONNECT = 0x02,
    LVDS_VSYNC_BUTT
} lvds_vsync_type_t;

typedef struct {
    lvds_vsync_type_t sync_type;

    /* hconnect vsync blanking len, valid when the sync_type is LVDS_VSYNC_HCONNECT */
    unsigned short hblank1;
    unsigned short hblank2;
} lvds_vsync_attr_t;

typedef enum {
    LVDS_FID_NONE = 0x00,
    LVDS_FID_IN_SAV = 0x01,  /* frame identification id in SAV 4th */
    LVDS_FID_IN_DATA = 0x02, /* frame identification id in first data */
    LVDS_FID_BUTT
} lvds_fid_type_t;

typedef struct {
    lvds_fid_type_t fid_type;

    /* Sony DOL has the Frame Information Line, in DOL H-Connection mode,
       should configure this flag as false to disable output the Frame Information Line */
    unsigned char output_fil;
} lvds_fid_attr_t;

typedef enum {
    LVDS_ENDIAN_LITTLE = 0x0,
    LVDS_ENDIAN_BIG = 0x1,
    LVDS_ENDIAN_BUTT
} lvds_bit_endian_t;

typedef struct {
    data_type_t input_data_type; /* data type: 8/10/12/14 bit */
    lvds_wdr_mode_t wdr_mode;         /* WDR mode */

    lvds_sync_mode_t sync_mode;   /* sync mode: SOF, SAV */
    lvds_vsync_attr_t vsync_attr; /* normal, share, hconnect */
    lvds_fid_attr_t fid_attr;     /* frame identification code */

    lvds_bit_endian_t data_endian;      /* data endian: little/big */
    lvds_bit_endian_t sync_code_endian; /* sync code endian: little/big */
    short lane_id[LVDS_LANE_NUM];       /* lane_id: -1 - disable */

    /* each vc has 4 params, sync_code[i]:
       sync_mode is SYNC_MODE_SOF: SOF, EOF, SOL, EOL
       sync_mode is SYNC_MODE_SAV: invalid sav, invalid eav, valid sav, valid eav  */
    unsigned short sync_code[LVDS_LANE_NUM][WDR_VC_NUM][SYNC_CODE_NUM];
} lvds_dev_attr_t;
typedef enum {
    SLVS_LANE_RATE_LOW = 0,         /* 1152Mbps */
    SLVS_LANE_RATE_HIGH = 1,        /* 2304Mbps */
    SLVS_LANE_RATE_BUTT
} slvs_lane_rate_t;
typedef enum {
    SLVS_ERR_CHECK_MODE_NONE = 0x0,       /* disable ECC & CRC */
    SLVS_ERR_CHECK_MODE_CRC = 0x1,        /* enable   CRC */
    SLVS_ERR_CHECK_MODE_ECC_2BYTE = 0x2,  /* enable   2 Byte ECC */
    SLVS_ERR_CHECK_MODE_ECC_4BYTE = 0x3,  /* enable   4 Byte ECC */

    SLVS_ERR_CHECK_MODE_BUTT
} slvs_err_check_mode_t;
typedef struct {
    data_type_t           input_data_type;          /* data type: 8/10/12/14/16 bit */
    slvs_wdr_mode_t       wdr_mode;                 /* WDR mode */
    slvs_lane_rate_t      lane_rate;
    hi_s32                   sensor_valid_width;
    short                 lane_id[SLVS_LANE_NUM];   /* lane_id: -1 - disable */
    slvs_err_check_mode_t err_check_mode;           /* ECC CRC mode */
} slvs_dev_attr_t;

typedef enum {
    SENSOR_CLK_74P25MHz  = 0x00,
    SENSOR_CLK_72MHz     = 0x01,
    SENSOR_CLK_54MHz     = 0x02,
    SENSOR_CLK_50MHz     = 0x03,
    SENSOR_CLK_24MHz     = 0x04,
    SENSOR_CLK_37P125MHz = 0x05,
    SENSOR_CLK_36MHz     = 0x06,
    SENSOR_CLK_27MHz     = 0x07,
    SENSOR_CLK_25MHz     = 0x08,
    SENSOR_CLK_12MHz     = 0x09,
    SENSOR_CLK_FREQ_BUTT
} sns_clk_freq_t;

typedef hi_u32 sns_clk_source_t;
typedef struct {
    sns_clk_source_t clk_source;
    sns_clk_freq_t clk_freq;
} sns_clk_cfg_t;

typedef hi_u32 sns_rst_source_t;

typedef hi_u32 combo_dev_t;
typedef struct {
    combo_dev_t devno;       /* device number */
    input_mode_t input_mode; /* input mode: MIPI/SUBLVDS/HISPI/SLVS_EC */
    mipi_data_rate_t data_rate;
    img_rect_t img_rect; /* MIPI Rx device crop area (corresponding to the original sensor input image size) */

    union {
        mipi_dev_attr_t mipi_attr; /* attribute of MIPI interface. AUTO:input_mode_t:INPUT_MODE_MIPI; */
        lvds_dev_attr_t lvds_attr; /* attribute of MIPI interface. AUTO:input_mode_t:INPUT_MODE_LVDS; */
        slvs_dev_attr_t slvs_attr;
    };
} combo_dev_attr_t;

#define HI_MIPI_IOC_MAGIC 'm'

/* init data lane, input mode, data type */
#define HI_MIPI_SET_DEV_ATTR                _IOW(HI_MIPI_IOC_MAGIC, 0x01, combo_dev_attr_t)

/* reset sensor */
#define HI_MIPI_RESET_SENSOR                _IOW(HI_MIPI_IOC_MAGIC, 0x05, sns_rst_source_t)

/* unreset sensor */
#define HI_MIPI_UNRESET_SENSOR              _IOW(HI_MIPI_IOC_MAGIC, 0x06, sns_rst_source_t)

/* reset mipi */
#define HI_MIPI_RESET_MIPI                  _IOW(HI_MIPI_IOC_MAGIC, 0x07, combo_dev_t)

/* unreset mipi */
#define HI_MIPI_UNRESET_MIPI                _IOW(HI_MIPI_IOC_MAGIC, 0x08, combo_dev_t)

/* set mipi hs_mode */
#define HI_MIPI_SET_HS_MODE                 _IOW(HI_MIPI_IOC_MAGIC, 0x0b, lane_divide_mode_t)

/* enable mipi clock */
#define HI_MIPI_ENABLE_MIPI_CLOCK           _IOW(HI_MIPI_IOC_MAGIC, 0x0c, combo_dev_t)

/* disable mipi clock */
#define HI_MIPI_DISABLE_MIPI_CLOCK          _IOW(HI_MIPI_IOC_MAGIC, 0x0d, combo_dev_t)

/* enable sensor clock */
#define HI_MIPI_ENABLE_SENSOR_CLOCK         _IOW(HI_MIPI_IOC_MAGIC, 0x10, sns_clk_source_t)

/* disable sensor clock */
#define HI_MIPI_DISABLE_SENSOR_CLOCK        _IOW(HI_MIPI_IOC_MAGIC, 0x11, sns_clk_source_t)

/* reset mipi */
#define HI_MIPI_RESET_SLVS                  _IOW(HI_MIPI_IOC_MAGIC, 0x9, combo_dev_t)

/* unreset mipi */
#define HI_MIPI_UNRESET_SLVS               _IOW(HI_MIPI_IOC_MAGIC, 0xa, combo_dev_t)

/* enable mipi clock */
#define HI_MIPI_ENABLE_SLVS_CLOCK           _IOW(HI_MIPI_IOC_MAGIC, 0xe, combo_dev_t)

/* disable mipi clock */
#define HI_MIPI_DISABLE_SLVS_CLOCK          _IOW(HI_MIPI_IOC_MAGIC, 0xf, combo_dev_t)

/* config sensor clock */
#define HI_MIPI_CONFIG_SENSOR_CLOCK        _IOW(HI_MIPI_IOC_MAGIC, 0x17, sns_clk_cfg_t)

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* OT_MIPI_RX_H */
