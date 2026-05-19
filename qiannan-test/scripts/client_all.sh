#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

bash qiannan-test/scripts/client_A_tp8.sh T1
bash qiannan-test/scripts/client_A_tp8.sh T6

# B
bash qiannan-test/scripts/client_B_dpa_tp8_dp8_router.sh T1
bash qiannan-test/scripts/client_B_dpa_tp8_dp8_router.sh T6

# C
bash qiannan-test/scripts/client_C_dpa_tp8_dp8_no_router.sh T1
bash qiannan-test/scripts/client_C_dpa_tp8_dp8_no_router.sh T6

# D
bash qiannan-test/scripts/client_D_dpa_tp4_dp2_router.sh T1
bash qiannan-test/scripts/client_D_dpa_tp4_dp2_router.sh T6
