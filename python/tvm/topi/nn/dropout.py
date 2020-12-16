# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-variable, unused-argument
"""Dropout operator."""

import tvm
from tvm import tir, topi


def dropout(data, gen, rate):
    rescale = tir.const(1 / (1 - rate), dtype=data.dtype)
    rate = tir.const(rate, dtype="float64")
    rands = topi.generic.threefry_uniform(gen, data.shape, 0.0, 1.0)
    mask = topi.cast(topi.greater_equal(rands, rate), dtype=data.dtype)
    return [topi.multiply(topi.multiply(data, mask), rescale), mask]
