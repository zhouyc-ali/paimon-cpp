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

#include "paimon/format/vortex/vortex_predicate_converter.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "arrow/type.h"
#include "paimon/common/utils/checked_cast.h"
#include "paimon/data/decimal.h"
#include "paimon/data/timestamp.h"
#include "paimon/defs.h"
#include "paimon/predicate/compound_predicate.h"
#include "paimon/predicate/function.h"
#include "paimon/predicate/leaf_predicate.h"
#include "paimon/predicate/literal.h"
#include "paimon/predicate/predicate.h"

namespace paimon::vortex {
namespace {

// A null VxExpressionPtr means "not pushable". Every failure path returns one, so the caller omits
// the scan filter and the framework's residual filter still applies the full predicate.
VxExpressionPtr NullExpr() {
    return VxExpressionPtr(nullptr, vx_expression_free);
}

// Free a Vortex error if one was set; a failed literal build just means "do not push this
// predicate".
void ClearError(vx_error* error) {
    if (error != nullptr) {
        vx_error_free(error);
    }
}

// vx_expression_not returns an owned handle typed const; reclaim it as an owned non-const pointer.
VxExpressionPtr NotOwned(const vx_expression* child) {
    return VxExpressionPtr(const_cast<vx_expression*>(vx_expression_not(child)),
                           vx_expression_free);
}

// Map a literal to a Vortex scalar expression, or null when its type is not pushed. `arrow_type` is
// the stored column type, needed only for TIMESTAMP (its unit and zone are not on the literal).
VxExpressionPtr MakeLiteralExpr(const Literal& literal,
                                const std::shared_ptr<arrow::DataType>& arrow_type) {
    if (literal.IsNull()) {
        return NullExpr();
    }
    VxScalarPtr scalar(nullptr, vx_scalar_free);
    vx_error* error = nullptr;
    switch (literal.GetType()) {
        case FieldType::BOOLEAN:
            scalar.reset(vx_scalar_new_bool(literal.GetValue<bool>(), false));
            break;
        case FieldType::TINYINT:
            scalar.reset(vx_scalar_new_i8(literal.GetValue<int8_t>(), false));
            break;
        case FieldType::SMALLINT:
            scalar.reset(vx_scalar_new_i16(literal.GetValue<int16_t>(), false));
            break;
        case FieldType::INT:
            scalar.reset(vx_scalar_new_i32(literal.GetValue<int32_t>(), false));
            break;
        // Vortex stores a date column as a date[days] extension and rejects comparing it against a
        // plain int scalar, so build a date-typed scalar.
        case FieldType::DATE:
            scalar.reset(vx_scalar_new_date(literal.GetValue<int32_t>(), false, &error));
            break;
        case FieldType::BIGINT:
            scalar.reset(vx_scalar_new_i64(literal.GetValue<int64_t>(), false));
            break;
        case FieldType::FLOAT:
            scalar.reset(vx_scalar_new_f32(literal.GetValue<float>(), false));
            break;
        case FieldType::DOUBLE:
            scalar.reset(vx_scalar_new_f64(literal.GetValue<double>(), false));
            break;
        case FieldType::STRING: {
            const auto value = literal.GetValue<std::string>();
            scalar.reset(vx_scalar_new_utf8(value.data(), value.size(), false, &error));
            break;
        }
        case FieldType::DECIMAL: {
            const auto decimal = literal.GetValue<Decimal>();
            const auto precision = static_cast<uint8_t>(decimal.Precision());
            const auto scale = static_cast<int8_t>(decimal.Scale());
            if (decimal.IsCompact()) {
                scalar.reset(vx_scalar_new_decimal_i64(decimal.ToUnscaledLong(), precision, scale,
                                                       false, &error));
            } else {
                // Little-endian 16-byte two's complement, low word first.
                uint8_t bytes16[16];
                const uint64_t low = decimal.LowBits();
                const uint64_t high = decimal.HighBits();
                std::memcpy(bytes16, &low, sizeof(low));
                std::memcpy(bytes16 + sizeof(low), &high, sizeof(high));
                scalar.reset(
                    vx_scalar_new_decimal_i128_le(bytes16, precision, scale, false, &error));
            }
            break;
        }
        case FieldType::TIMESTAMP: {
            if (arrow_type == nullptr || arrow_type->id() != arrow::Type::TIMESTAMP) {
                return NullExpr();
            }
            const auto& ts_type = checked_cast<const arrow::TimestampType&>(*arrow_type);
            const auto ts = literal.GetValue<Timestamp>();
            int64_t value = 0;
            vx_time_unit unit = VX_TIME_UNIT_MILLISECOND;
            switch (ts_type.unit()) {
                case arrow::TimeUnit::SECOND:
                case arrow::TimeUnit::MILLI:
                    value = ts.GetMillisecond();
                    unit = VX_TIME_UNIT_MILLISECOND;
                    break;
                case arrow::TimeUnit::MICRO:
                    value = ts.ToMicrosecond();
                    unit = VX_TIME_UNIT_MICROSECOND;
                    break;
                case arrow::TimeUnit::NANO:
                    value = ts.ToNanosecond();
                    unit = VX_TIME_UNIT_NANOSECOND;
                    break;
            }
            // A stored zone marks a local-zoned timestamp; paimon-java pushes those as UTC
            // instants.
            const char* zone = ts_type.timezone().empty() ? nullptr : "UTC";
            scalar.reset(vx_scalar_new_timestamp(value, unit, zone, false, &error));
            break;
        }
        default:
            // BINARY, TIME and nested types are not pushed (aligned with paimon-java).
            return NullExpr();
    }
    ClearError(error);
    if (scalar == nullptr) {
        return NullExpr();
    }
    vx_error* literal_error = nullptr;
    vx_expression* expr = vx_expression_literal(scalar.get(), &literal_error);
    ClearError(literal_error);
    return VxExpressionPtr(expr, vx_expression_free);
}

// The Vortex operator for a comparison function, or false when the function is not a comparison.
bool ComparisonOperator(Function::Type type, vx_binary_operator* op) {
    switch (type) {
        case Function::Type::EQUAL:
            *op = VX_OPERATOR_EQ;
            return true;
        case Function::Type::NOT_EQUAL:
            *op = VX_OPERATOR_NOT_EQ;
            return true;
        case Function::Type::GREATER_THAN:
            *op = VX_OPERATOR_GT;
            return true;
        case Function::Type::GREATER_OR_EQUAL:
            *op = VX_OPERATOR_GTE;
            return true;
        case Function::Type::LESS_THAN:
            *op = VX_OPERATOR_LT;
            return true;
        case Function::Type::LESS_OR_EQUAL:
            *op = VX_OPERATOR_LTE;
            return true;
        default:
            return false;
    }
}

VxExpressionPtr ConvertLeaf(const std::shared_ptr<LeafPredicate>& leaf,
                            const std::shared_ptr<arrow::Schema>& file_schema,
                            const vx_expression* root) {
    const Function::Type type = leaf->GetFunction().GetType();
    VxExpressionPtr column(vx_expression_get_item(leaf->FieldName().c_str(), root),
                           vx_expression_free);
    if (column == nullptr) {
        return NullExpr();
    }
    if (type == Function::Type::IS_NULL) {
        return VxExpressionPtr(vx_expression_is_null(column.get()), vx_expression_free);
    }
    if (type == Function::Type::IS_NOT_NULL) {
        VxExpressionPtr is_null(vx_expression_is_null(column.get()), vx_expression_free);
        if (is_null == nullptr) {
            return NullExpr();
        }
        return NotOwned(is_null.get());
    }
    vx_binary_operator op = VX_OPERATOR_EQ;
    if (!ComparisonOperator(type, &op)) {
        return NullExpr();  // IN / NOT_IN / STARTS_WITH / ... are not pushed.
    }
    if (leaf->Literals().empty()) {
        return NullExpr();
    }
    std::shared_ptr<arrow::DataType> arrow_type;
    const int field_index =
        file_schema != nullptr ? file_schema->GetFieldIndex(leaf->FieldName()) : /*not found=*/-1;
    if (field_index >= 0) {
        arrow_type = file_schema->field(field_index)->type();
    }
    VxExpressionPtr literal_expr = MakeLiteralExpr(leaf->Literals()[0], arrow_type);
    if (literal_expr == nullptr) {
        return NullExpr();
    }
    return VxExpressionPtr(vx_expression_binary(op, column.get(), literal_expr.get()),
                           vx_expression_free);
}

VxExpressionPtr ConvertImpl(const std::shared_ptr<Predicate>& predicate,
                            const std::shared_ptr<arrow::Schema>& file_schema,
                            const vx_expression* root) {
    if (predicate == nullptr) {
        return NullExpr();
    }
    const Function::Type type = predicate->GetFunction().GetType();
    if (type == Function::Type::AND || type == Function::Type::OR) {
        const auto compound = std::dynamic_pointer_cast<CompoundPredicate>(predicate);
        if (compound == nullptr) {
            return NullExpr();
        }
        std::vector<VxExpressionPtr> children;
        children.reserve(compound->Children().size());
        for (const std::shared_ptr<Predicate>& child : compound->Children()) {
            VxExpressionPtr converted = ConvertImpl(child, file_schema, root);
            if (converted != nullptr) {
                children.push_back(std::move(converted));
            } else if (type == Function::Type::OR) {
                return NullExpr();  // OR is all-or-nothing: one unpushable child rejects the whole.
            }
            // AND: an unpushable child is dropped; the residual filter still applies it.
        }
        if (children.empty()) {
            return NullExpr();
        }
        std::vector<const vx_expression*> raw;
        raw.reserve(children.size());
        for (const VxExpressionPtr& child : children) {
            raw.push_back(child.get());
        }
        vx_expression* combined = type == Function::Type::AND
                                      ? vx_expression_and(raw.data(), raw.size())
                                      : vx_expression_or(raw.data(), raw.size());
        return VxExpressionPtr(combined, vx_expression_free);
    }
    const auto leaf = std::dynamic_pointer_cast<LeafPredicate>(predicate);
    if (leaf == nullptr) {
        return NullExpr();
    }
    return ConvertLeaf(leaf, file_schema, root);
}

}  // namespace

VxExpressionPtr VortexPredicateConverter::Convert(
    const std::shared_ptr<Predicate>& predicate,
    const std::shared_ptr<arrow::Schema>& file_schema) {
    if (predicate == nullptr) {
        return NullExpr();
    }
    VxExpressionPtr root(vx_expression_root(), vx_expression_free);
    if (root == nullptr) {
        return NullExpr();
    }
    return ConvertImpl(predicate, file_schema, root.get());
}

}  // namespace paimon::vortex
