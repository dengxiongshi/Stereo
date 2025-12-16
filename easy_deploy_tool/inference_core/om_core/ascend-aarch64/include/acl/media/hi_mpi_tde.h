/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: tde API header file
 * Author: Hisilicon multimedia software group
 * Create: 2023/03/10
 */

#ifndef HI_MPI_TDE_H
#define HI_MPI_TDE_H

#include "hi_base_tde.h"

#ifdef __cplusplus
extern "C" {
#endif

/* API Declaration */
hi_s32 hi_tde_open(hi_void);
hi_void hi_tde_close(hi_void);
hi_s32 hi_tde_begin_job(hi_void);
hi_s32 hi_tde_end_job(hi_s32 handle, hi_bool is_sync, hi_bool is_block, hi_u32 time_out);
hi_s32 hi_tde_cancel_job(hi_s32 handle);
hi_s32 hi_tde_wait_for_done(hi_s32 handle);
hi_s32 hi_tde_wait_all_done(hi_void);
hi_s32 hi_tde_quick_fill(hi_s32 handle, const hi_tde_none_src *none_src, hi_u32 fill_data);
hi_s32 hi_tde_quick_copy(hi_s32 handle, const hi_tde_single_src *single_src);
hi_s32 hi_tde_pattern_fill(hi_s32 handle, const hi_tde_double_src *double_src, const hi_tde_pattern_fill_opt *fill_opt);

#ifdef __cplusplus
}
#endif
#endif /* HI_MPI_TDE_H */
