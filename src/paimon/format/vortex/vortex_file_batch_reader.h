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
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "arrow/type_fwd.h"
#include "paimon/format/vortex/vortex_ffi.h"
#include "paimon/format/vortex/vortex_ffi_util.h"
#include "paimon/format/vortex/vortex_io_callbacks.h"
#include "paimon/reader/file_batch_reader.h"
#include "paimon/result.h"

namespace arrow {
class Array;
class DataType;
class MemoryPool;
class Schema;
}  // namespace arrow

namespace paimon {
class InputStream;
class Metrics;
class PredicateFilter;
}  // namespace paimon

namespace paimon::vortex {

/// Reads a Vortex file via the vortex-ffi scan API.
class VortexFileBatchReader : public FileBatchReader {
 public:
    static Result<std::unique_ptr<VortexFileBatchReader>> Create(
        const std::shared_ptr<InputStream>& input, int32_t batch_size,
        const std::shared_ptr<arrow::MemoryPool>& arrow_pool);

    ~VortexFileBatchReader() override;

    Result<ReadBatch> NextBatch() override;
    Result<std::unique_ptr<::ArrowSchema>> GetFileSchema() const override;
    Status SetReadSchema(::ArrowSchema* read_schema, const std::shared_ptr<Predicate>& predicate,
                         const std::optional<RoaringBitmap32>& selection_bitmap) override;
    Result<uint64_t> GetPreviousBatchFileRowId(uint64_t batch_row_id) const override;
    Result<uint64_t> GetNumberOfRows() const override;
    std::shared_ptr<Metrics> GetReaderMetrics() const override;
    void Close() override;
    bool SupportPreciseBitmapSelection() const override {
        return true;
    }

 private:
    VortexFileBatchReader(const std::shared_ptr<InputStream>& input, int32_t batch_size,
                          std::shared_ptr<VortexInputContext> input_context, VxSessionPtr session,
                          VxDataSourcePtr data_source, VxScanPtr scan,
                          const std::shared_ptr<arrow::Schema>& file_schema, uint64_t total_rows,
                          const std::shared_ptr<arrow::MemoryPool>& arrow_pool);

    /// Pull the next Arrow array from the current partition stream, opening the next partition's
    /// stream when the current one is exhausted. Returns nullptr at end of scan.
    Result<std::shared_ptr<arrow::Array>> ReadNextArray();
    /// Open the next partition and fill `current_stream_` via `vx_partition_scan_arrow`.
    /// Returns false when there are no more partitions.
    Result<bool> OpenNextPartitionStream();
    /// Reshape a scanned batch to `read_schema_`, selecting columns by name and recursing into
    /// nested fields. Returns `array` unchanged when SetReadSchema has not been called.
    Result<std::shared_ptr<arrow::Array>> AlignToReadSchema(
        const std::shared_ptr<arrow::Array>& array) const;
    void ReleaseStream();
    void CloseInternal();

    // Teardown order matters, so these are declared to be destroyed in a safe reverse order and
    // also reset explicitly in CloseInternal(): scan_ borrows data_source_, data_source_ reads
    // through input_context_, and the Arrow stream must be released while session_ is alive.
    std::shared_ptr<InputStream> input_;
    int32_t batch_size_;
    std::shared_ptr<VortexInputContext> input_context_;
    VxSessionPtr session_;
    VxDataSourcePtr data_source_;
    VxScanPtr scan_;
    // The full file schema as exported by Vortex (view types already normalized to standard); this
    // is what GetFileSchema() reports to paimon.
    std::shared_ptr<arrow::Schema> file_schema_;
    // The schema NextBatch must output, set by SetReadSchema (the projected read schema). Null
    // until SetReadSchema is called, in which case the full file schema is returned.
    std::shared_ptr<arrow::Schema> read_schema_;
    uint64_t total_rows_ = 0;

    ::ArrowArrayStream current_stream_{};
    // The scan's actual output schema for current_stream_ (projected columns, standard types);
    // ReadNextArray imports each batch as this struct type.
    std::shared_ptr<arrow::Schema> current_stream_schema_;
    bool stream_active_ = false;
    std::shared_ptr<arrow::Array> current_batch_;
    int64_t current_batch_offset_ = 0;

    uint64_t rows_emitted_ = 0;
    uint64_t previous_first_row_ = std::numeric_limits<uint64_t>::max();
    uint64_t previous_batch_row_count_ = 0;

    // Row selection pushed into the scan as include-by-index. selection_row_ids_ holds the selected
    // original row ids in ascending order; GetPreviousBatchFileRowId maps an emitted row back to
    // its physical position through it (Vortex returns selected rows in ascending order under
    // ordered).
    bool has_selection_ = false;
    std::vector<uint64_t> selection_row_ids_;

    // Predicate pushed into the scan as a native filter. Filtering drops non-matching rows, so the
    // scan also returns a hidden `_row_idx` column (include_row_idx) with each surviving row's
    // original physical position; previous_batch_row_idx_ holds that column for the last batch and
    // GetPreviousBatchFileRowId reads it, since the rows_emitted_ accumulator is wrong once rows
    // are dropped.
    bool has_filter_ = false;
    std::shared_ptr<arrow::Array> previous_batch_row_idx_;

    std::shared_ptr<arrow::MemoryPool> arrow_pool_;
    std::shared_ptr<PredicateFilter>
        predicate_filter_;  // from SetReadSchema; pushed down if convertible
    std::shared_ptr<Metrics> metrics_;
    bool closed_ = false;
};

}  // namespace paimon::vortex
