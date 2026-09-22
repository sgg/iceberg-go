// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

package table

import (
	"context"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/compute"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/iceberg-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func identSort(sourceID int, dir SortDirection, null NullOrder) SortField {
	return SortField{
		SourceIDs: []int{sourceID}, Transform: iceberg.IdentityTransform{},
		Direction: dir, NullOrder: null,
	}
}

func TestResolveSortKeys(t *testing.T) {
	for _, tt := range []struct {
		name        string
		schema      *iceberg.Schema
		unsorted    bool
		fields      []SortField
		want        []sortKey
		errContains string
	}{
		{
			name:     "unsorted",
			schema:   iceberg.NewSchema(0, iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64, Required: true}),
			unsorted: true,
		},
		{
			name: "direction and null order",
			schema: iceberg.NewSchema(0,
				iceberg.NestedField{ID: 1, Name: "a", Type: iceberg.PrimitiveTypes.Int64, Required: false},
				iceberg.NestedField{ID: 2, Name: "b", Type: iceberg.PrimitiveTypes.String, Required: false},
			),
			fields: []SortField{identSort(2, SortDESC, NullsFirst), identSort(1, SortASC, NullsLast)},
			want: []sortKey{
				{path: []int{1}, SortKey: compute.SortKey{ColumnIndex: 1, Order: compute.SortOrderDescending, NullPlacement: compute.SortNullsAtStart}},
				{path: []int{0}, SortKey: compute.SortKey{ColumnIndex: 0, Order: compute.SortOrderAscending, NullPlacement: compute.SortNullsAtEnd}},
			},
		},
		{
			name: "nested struct field",
			schema: iceberg.NewSchema(0,
				iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64, Required: true},
				iceberg.NestedField{ID: 2, Name: "loc", Type: &iceberg.StructType{FieldList: []iceberg.NestedField{
					{ID: 3, Name: "lat", Type: iceberg.PrimitiveTypes.Float64, Required: false},
					{ID: 4, Name: "lon", Type: iceberg.PrimitiveTypes.Float64, Required: false},
				}}, Required: false},
			),
			fields: []SortField{identSort(3, SortDESC, NullsFirst)},
			want: []sortKey{
				{path: []int{1, 0}, SortKey: compute.SortKey{ColumnIndex: 1, Order: compute.SortOrderDescending, NullPlacement: compute.SortNullsAtStart}},
			},
		},
		{
			name: "deeply nested struct field",
			schema: iceberg.NewSchema(0,
				iceberg.NestedField{ID: 1, Name: "outer", Type: &iceberg.StructType{FieldList: []iceberg.NestedField{
					{ID: 2, Name: "inner", Type: &iceberg.StructType{FieldList: []iceberg.NestedField{
						{ID: 3, Name: "val", Type: iceberg.PrimitiveTypes.Int64, Required: true},
					}}, Required: true},
				}}, Required: true},
			),
			fields: []SortField{identSort(3, SortASC, NullsLast)},
			want: []sortKey{
				{path: []int{0, 0, 0}, SortKey: compute.SortKey{ColumnIndex: 0, Order: compute.SortOrderAscending, NullPlacement: compute.SortNullsAtEnd}},
			},
		},
		{
			name: "missing source id",
			schema: iceberg.NewSchema(0,
				iceberg.NestedField{ID: 1, Name: "a", Type: iceberg.PrimitiveTypes.Int64, Required: true},
			),
			fields:      []SortField{identSort(99, SortASC, NullsFirst)},
			errContains: "source id 99",
		},
		{
			name: "list element rejected",
			schema: iceberg.NewSchema(0,
				iceberg.NestedField{ID: 1, Name: "ids", Type: &iceberg.ListType{
					ElementID: 2, Element: iceberg.PrimitiveTypes.Int64, ElementRequired: true,
				}, Required: true},
			),
			fields:      []SortField{identSort(2, SortASC, NullsLast)},
			errContains: "list or map",
		},
		{
			name: "map value rejected",
			schema: iceberg.NewSchema(0,
				iceberg.NestedField{ID: 1, Name: "props", Type: &iceberg.MapType{
					KeyID: 2, KeyType: iceberg.PrimitiveTypes.String,
					ValueID: 3, ValueType: iceberg.PrimitiveTypes.Int64, ValueRequired: false,
				}, Required: true},
			),
			fields:      []SortField{identSort(3, SortASC, NullsLast)},
			errContains: "list or map",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			order := UnsortedSortOrder
			if !tt.unsorted {
				var err error
				order, err = NewSortOrder(1, tt.fields)
				require.NoError(t, err)
			}

			keys, err := resolveSortKeys(order, tt.schema)
			if tt.errContains != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.errContains)

				return
			}

			require.NoError(t, err)
			assert.Equal(t, tt.want, keys)
		})
	}
}

type locRow struct {
	id    int32
	lat   int32
	locOK bool
}

func locBatch(t *testing.T, mem memory.Allocator, rows []locRow) arrow.RecordBatch {
	t.Helper()
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int32, Nullable: true},
		{Name: "loc", Type: arrow.StructOf(
			arrow.Field{Name: "lat", Type: arrow.PrimitiveTypes.Int32, Nullable: true},
		), Nullable: true},
	}, nil)
	bldr := array.NewRecordBuilder(mem, schema)
	idBldr := bldr.Field(0).(*array.Int32Builder)
	stBldr := bldr.Field(1).(*array.StructBuilder)
	latBldr := stBldr.FieldBuilder(0).(*array.Int32Builder)
	for _, r := range rows {
		idBldr.Append(r.id)
		stBldr.Append(r.locOK)
		latBldr.Append(r.lat)
	}
	batch := bldr.NewRecordBatch()
	bldr.Release()

	return batch
}

func assertSortedIDs(t *testing.T, ctx context.Context, batch arrow.RecordBatch, keys []sortKey, want []int32) {
	t.Helper()
	sorted, err := applySortKeys(ctx, batch, keys)
	require.NoError(t, err)
	defer sorted.Release()

	ids := sorted.Column(0).(*array.Int32)
	got := make([]int32, ids.Len())
	for i := range got {
		got[i] = ids.Value(i)
	}
	assert.Equal(t, want, got)
}

func TestApplySortKeys(t *testing.T) {
	mem := memory.DefaultAllocator
	ctx := compute.WithAllocator(t.Context(), mem)

	nullParentRows := []locRow{
		{id: 1, lat: 20, locOK: true},
		{id: 2, lat: 0, locOK: false}, // lat 0 under a null parent must sort as null, not 0
		{id: 3, lat: 10, locOK: true},
	}

	for _, tt := range []struct {
		name         string
		rows         []locRow
		nullsAtStart bool
		wantIDs      []int32
	}{
		{
			name:    "nested leaf",
			rows:    []locRow{{id: 1, lat: 30, locOK: true}, {id: 2, lat: 10, locOK: true}, {id: 3, lat: 20, locOK: true}},
			wantIDs: []int32{2, 3, 1},
		},
		{
			name:    "null ancestor/nulls last",
			rows:    nullParentRows,
			wantIDs: []int32{3, 1, 2},
		},
		{
			name:         "null ancestor/nulls first",
			rows:         nullParentRows,
			nullsAtStart: true,
			wantIDs:      []int32{2, 3, 1},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			batch := locBatch(t, mem, tt.rows)
			defer batch.Release()
			key := sortKey{
				path: []int{1, 0},
				SortKey: compute.SortKey{
					Order: compute.SortOrderAscending, NullPlacement: compute.SortNullsAtEnd,
				},
			}
			if tt.nullsAtStart {
				key.NullPlacement = compute.SortNullsAtStart
			}
			assertSortedIDs(t, ctx, batch, []sortKey{key}, tt.wantIDs)
		})
	}

	t.Run("top-level fast path", func(t *testing.T) {
		schema := arrow.NewSchema([]arrow.Field{
			{Name: "id", Type: arrow.PrimitiveTypes.Int32, Nullable: true},
		}, nil)
		bldr := array.NewRecordBuilder(mem, schema)
		idBldr := bldr.Field(0).(*array.Int32Builder)
		for _, v := range []int32{3, 1, 2} {
			idBldr.Append(v)
		}
		batch := bldr.NewRecordBatch()
		bldr.Release()
		defer batch.Release()

		assertSortedIDs(t, ctx, batch, []sortKey{{
			path:    []int{0},
			SortKey: compute.SortKey{ColumnIndex: 0, Order: compute.SortOrderAscending, NullPlacement: compute.SortNullsAtEnd},
		}}, []int32{1, 2, 3})
	})
}
