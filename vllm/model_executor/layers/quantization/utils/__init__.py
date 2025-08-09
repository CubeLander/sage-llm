# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .layer_utils import replace_parameter
from .layer_utils import update_tensor_inplace

__all__ = ['update_tensor_inplace', 'replace_parameter']
