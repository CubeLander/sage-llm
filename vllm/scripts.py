# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.cli.main import main as vllm_main
from hotLLM.vllm.logger import init_logger

logger = init_logger(__name__)


# Backwards compatibility for the move from vllm.scripts to
# vllm.cli.main
def main():
    logger.warning("vllm.scripts.main() is deprecated. Please re-install "
                   "vllm or use vllm.cli.main.main() instead.")
    vllm_main()
