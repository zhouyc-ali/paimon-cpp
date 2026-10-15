/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <memory>

#include "paimon/format/vortex/vortex_ffi_util.h"

namespace arrow {
class Schema;
}  // namespace arrow

namespace paimon {
class Predicate;
}  // namespace paimon

namespace paimon::vortex {

/// Converts a paimon `Predicate` into a Vortex filter expression, mirroring paimon-java's
/// `VortexPredicateConverter`. Returns a null pointer when the predicate cannot be pushed down (an
/// unsupported function or literal type); the caller then omits the scan filter and relies on the
/// framework's residual filter. `file_schema` supplies the stored timestamp unit and zone, which
/// the paimon `Literal` does not carry.
class VortexPredicateConverter {
 public:
    VortexPredicateConverter() = delete;
    ~VortexPredicateConverter() = delete;

    /// Build the Vortex filter expression for `predicate`, or a null pointer if it cannot be
    /// pushed.
    static VxExpressionPtr Convert(const std::shared_ptr<Predicate>& predicate,
                                   const std::shared_ptr<arrow::Schema>& file_schema);
};

}  // namespace paimon::vortex
