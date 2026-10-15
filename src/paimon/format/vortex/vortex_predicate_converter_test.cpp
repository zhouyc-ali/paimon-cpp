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

#include <memory>
#include <vector>

#include "arrow/api.h"
#include "gtest/gtest.h"
#include "paimon/data/decimal.h"
#include "paimon/data/timestamp.h"
#include "paimon/defs.h"
#include "paimon/predicate/literal.h"
#include "paimon/predicate/predicate.h"
#include "paimon/predicate/predicate_builder.h"
#include "paimon/testing/utils/testharness.h"

namespace paimon::vortex {
namespace {

// Only timestamp leaves consult the schema (for the stored unit and zone); the other types are
// converted from the literal alone, so their field names need not appear here.
std::shared_ptr<arrow::Schema> TestSchema() {
    return arrow::schema({
        arrow::field("i", arrow::int32()),
        arrow::field("s", arrow::utf8()),
        arrow::field("ts_ms", arrow::timestamp(arrow::TimeUnit::MILLI)),
        arrow::field("ts_ltz", arrow::timestamp(arrow::TimeUnit::MICRO, "UTC")),
    });
}

bool Pushed(const std::shared_ptr<Predicate>& predicate,
            const std::shared_ptr<arrow::Schema>& schema) {
    return VortexPredicateConverter::Convert(predicate, schema) != nullptr;
}

TEST(VortexPredicateConverterTest, ComparisonsOnSupportedTypesArePushed) {
    const std::shared_ptr<arrow::Schema> schema = TestSchema();
    EXPECT_TRUE(
        Pushed(PredicateBuilder::Equal(0, "i", FieldType::INT, Literal(static_cast<int32_t>(42))),
               schema));
    EXPECT_TRUE(
        Pushed(PredicateBuilder::NotEqual(0, "i", FieldType::INT, Literal(static_cast<int32_t>(1))),
               schema));
    EXPECT_TRUE(Pushed(
        PredicateBuilder::GreaterThan(0, "l", FieldType::BIGINT, Literal(static_cast<int64_t>(7))),
        schema));
    EXPECT_TRUE(Pushed(
        PredicateBuilder::GreaterOrEqual(0, "i", FieldType::INT, Literal(static_cast<int32_t>(1))),
        schema));
    EXPECT_TRUE(
        Pushed(PredicateBuilder::LessThan(0, "i", FieldType::INT, Literal(static_cast<int32_t>(1))),
               schema));
    EXPECT_TRUE(
        Pushed(PredicateBuilder::LessOrEqual(0, "d", FieldType::DOUBLE, Literal(3.5)), schema));
    EXPECT_TRUE(Pushed(PredicateBuilder::Equal(0, "f", FieldType::FLOAT, Literal(1.5F)), schema));
    EXPECT_TRUE(Pushed(PredicateBuilder::Equal(0, "b", FieldType::BOOLEAN, Literal(true)), schema));
    EXPECT_TRUE(Pushed(
        PredicateBuilder::Equal(0, "s", FieldType::STRING, Literal(FieldType::STRING, "abc", 3)),
        schema));
    EXPECT_TRUE(
        Pushed(PredicateBuilder::Equal(0, "dt", FieldType::DATE, Literal(FieldType::DATE, 19000)),
               schema));
    // Compact decimal (precision <= 18) and wide decimal (precision > 18) both push.
    EXPECT_TRUE(Pushed(PredicateBuilder::Equal(0, "dec", FieldType::DECIMAL,
                                               Literal(Decimal::FromUnscaledLong(12345, 10, 2))),
                       schema));
    EXPECT_TRUE(Pushed(PredicateBuilder::Equal(0, "dec", FieldType::DECIMAL,
                                               Literal(Decimal::FromUnscaledLong(12345, 19, 2))),
                       schema));
}

TEST(VortexPredicateConverterTest, IsNullAndIsNotNullArePushed) {
    const std::shared_ptr<arrow::Schema> schema = TestSchema();
    EXPECT_TRUE(Pushed(PredicateBuilder::IsNull(0, "i", FieldType::INT), schema));
    EXPECT_TRUE(Pushed(PredicateBuilder::IsNotNull(0, "i", FieldType::INT), schema));
}

TEST(VortexPredicateConverterTest, TimestampUsesStoredColumnUnit) {
    const std::shared_ptr<arrow::Schema> schema = TestSchema();
    // Present in the schema, so the stored unit/zone is known: pushed.
    EXPECT_TRUE(Pushed(PredicateBuilder::Equal(0, "ts_ms", FieldType::TIMESTAMP,
                                               Literal(Timestamp::FromEpochMillis(1234))),
                       schema));
    EXPECT_TRUE(Pushed(PredicateBuilder::Equal(0, "ts_ltz", FieldType::TIMESTAMP,
                                               Literal(Timestamp::FromEpochMillis(1234))),
                       schema));
    // Absent from the schema, so the unit is unknown: not pushed.
    EXPECT_FALSE(Pushed(PredicateBuilder::Equal(0, "absent", FieldType::TIMESTAMP,
                                                Literal(Timestamp::FromEpochMillis(1234))),
                        schema));
}

TEST(VortexPredicateConverterTest, UnsupportedFunctionsAndLiteralsAreNotPushed) {
    const std::shared_ptr<arrow::Schema> schema = TestSchema();
    // IN is not pushed.
    EXPECT_FALSE(Pushed(
        PredicateBuilder::In(0, "i", FieldType::INT,
                             {Literal(static_cast<int32_t>(1)), Literal(static_cast<int32_t>(2))}),
        schema));
    // STARTS_WITH is not pushed.
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<Predicate> starts_with,
                         PredicateBuilder::StartsWith(0, "s", FieldType::STRING,
                                                      Literal(FieldType::STRING, "a", 1)));
    EXPECT_FALSE(Pushed(starts_with, schema));
    // A BINARY literal is not pushed (aligned with paimon-java).
    EXPECT_FALSE(Pushed(
        PredicateBuilder::Equal(0, "bin", FieldType::BINARY, Literal(FieldType::BINARY, "x", 1)),
        schema));
    // A null literal is not pushed.
    EXPECT_FALSE(
        Pushed(PredicateBuilder::Equal(0, "i", FieldType::INT, Literal(FieldType::INT)), schema));
}

TEST(VortexPredicateConverterTest, AndDropsUnsupportedChildrenButOrRejects) {
    const std::shared_ptr<arrow::Schema> schema = TestSchema();
    const std::shared_ptr<Predicate> supported =
        PredicateBuilder::Equal(0, "i", FieldType::INT, Literal(static_cast<int32_t>(1)));
    const std::shared_ptr<Predicate> other =
        PredicateBuilder::Equal(0, "i", FieldType::INT, Literal(static_cast<int32_t>(2)));
    const std::shared_ptr<Predicate> unsupported =
        PredicateBuilder::In(0, "i", FieldType::INT, {Literal(static_cast<int32_t>(1))});

    // AND drops the unsupported child and still pushes the supported one.
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<Predicate> and_partial,
                         PredicateBuilder::And({supported, unsupported}));
    EXPECT_TRUE(Pushed(and_partial, schema));

    // AND with nothing pushable yields null.
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<Predicate> and_none,
                         PredicateBuilder::And({unsupported, unsupported}));
    EXPECT_FALSE(Pushed(and_none, schema));

    // OR of supported children pushes.
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<Predicate> or_full,
                         PredicateBuilder::Or({supported, other}));
    EXPECT_TRUE(Pushed(or_full, schema));

    // OR is all-or-nothing: one unsupported child rejects the whole expression.
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<Predicate> or_partial,
                         PredicateBuilder::Or({supported, unsupported}));
    EXPECT_FALSE(Pushed(or_partial, schema));
}

TEST(VortexPredicateConverterTest, NullPredicateIsNotPushed) {
    EXPECT_FALSE(Pushed(nullptr, TestSchema()));
}

}  // namespace
}  // namespace paimon::vortex
