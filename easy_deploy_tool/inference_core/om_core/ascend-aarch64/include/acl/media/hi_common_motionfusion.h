/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2018-2020. All rights reserved.
 * Description : hi_comm_motionfusion.h
 * Author : ISP SW
 * Create : 2018-12-22
 */
#ifndef HI_COMMON_MOTIONFUSION_H
#define HI_COMMON_MOTIONFUSION_H

#include "hi_media_type.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* __cplusplus */

#define HI_ERR_MOTIONFUSION_SYS_NOTREADY  HI_DEF_ERR(HI_ID_MOTIONFUSION, EN_ERR_LEVEL_ERROR, EN_ERR_SYS_NOTREADY)
#define HI_ERR_MOTIONFUSION_NOT_PERMITTED HI_DEF_ERR(HI_ID_MOTIONFUSION, EN_ERR_LEVEL_ERROR, EN_ERR_NOT_PERM)
#define HI_ERR_MOTIONFUSION_INVALID_CHNID HI_DEF_ERR(HI_ID_MOTIONFUSION, EN_ERR_LEVEL_ERROR, EN_ERR_INVALID_CHNID)
#define HI_ERR_MOTIONFUSION_NULL_PTR      HI_DEF_ERR(HI_ID_MOTIONFUSION, EN_ERR_LEVEL_ERROR, EN_ERR_NULL_PTR)

typedef struct {
    /*
     * RW; continues steady time (in sec)
     * threshold for steady detection
     * range: [0, (1<<16-1]
     */
    hi_u32 steady_time_thr;
    /*
     * RW; max gyro ZRO tolerance presented in datasheet,
     * with (ADC word length - 1) decimal bits
     * range: [0, 100 * (1<<15)]
     */
    hi_s32 gyro_offset;
    /*
     * RW; max acc ZRO tolerance presented in datasheet,
     * with (ADC word length - 1) decimal bits
     * range: [0, 0.5 * (1<<15)]
     */
    hi_s32 acc_offset;
    /*
     * RW; gyro rms noise of under the current filter BW,
     * with (ADC Word Length - 1) decimal bits
     * range: [0, 0.5 * (1<<15)]
     */
    hi_s32 gyro_rms;
    /*
     * RW; acc rms noise of under the current filter BW
     * with (acc word length - 1) decimal bits
     * range: [0, 0.005 * (1<<15)]
     */
    hi_s32 acc_rms;
    /*
     * RW; scale factor of gyro offset for steady detection,
     * larger -> higher recall, but less the precision
     * range: [0, 1000 * (1<<4)]
     */
    hi_s32 gyro_offset_factor;
    /*
     * RW; scale factor of acc offset for steady detection,
     * larger -> higher recall, but less the precision
     * range: [0, 1000 * (1<<4)]
     */
    hi_s32 acc_offset_factor;
    /*
     * RW; scale factor of gyro rms for steady detection,
     * larger -> higher recall, but less the precision
     * range: [0, 1000 * (1<<4)]
     */
    hi_s32 gyro_rms_factor;
    /*
     * RW; scale factor of acc rms for steady detection,
     * larger -> higher recall, but less the precision
     * range: [0, 1000 * (1<<4)]
     */
    hi_s32 acc_rms_factor;
} hi_mfusion_steady_detect_attr;

typedef struct {
    hi_u32 device_mask;      /* device mask: gyro,acc or magn */
    hi_u32 temperature_mask; /* temperature mask: gyro temperature ,acc temperatureor magn temperature */
    hi_mfusion_steady_detect_attr steady_detect_attr;
} hi_mfusion_attr;

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */

#endif
