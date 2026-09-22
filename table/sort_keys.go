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
	"fmt"
	"slices"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/bitutil"
	"github.com/apache/arrow-go/v18/arrow/compute"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/iceberg-go"
	iceberginternal "github.com/apache/iceberg-go/internal"
)

// sortKey is a resolved sort field: a struct-index path from a record-batch
// column down to the source leaf, plus Arrow sort options. A path of length 1
// is a top-level column and can be passed to [compute.SortRecordBatch] as-is.
type sortKey struct {
	path []int
	compute.SortKey
}

// resolveSortKeys converts an Iceberg SortOrder into Arrow compute sort keys
// against the given file schema. Each sort field's source column is used as
// the sort key directly: order-preserving transforms (identity, truncate,
// year/month/day/hour) sort the same as their source values.
//
// Returns nil keys for an unsorted SortOrder. Returns an error when a sort
// field's source id is not in fileSchema, or is nested under a list or map.
//
// Limitations:
//
//   - Per-batch, not per-file: keys are applied to each batch independently,
//     not merged into a globally sorted run across batches. Files therefore
//     record no sort_order_id, which asserts the whole file is sorted.
//   - Bucket transform falls back to source-column order, since the hash
//     bucket is not order-preserving.
//   - Multi-argument transform sort fields use only the first source column.
//   - Sources nested under lists or maps are not supported.
func resolveSortKeys(order SortOrder, fileSchema *iceberg.Schema) ([]sortKey, error) {
	if order.IsUnsorted() {
		return nil, nil
	}

	fields := fileSchema.FieldsRef(iceberginternal.SchemaRef{})
	keys := make([]sortKey, 0, order.Len())
	for _, field := range order.fields {
		path, err := sortFieldPath(fileSchema, fields, field.SourceID(), order.OrderID())
		if err != nil {
			return nil, err
		}

		key := sortKey{
			path:    path,
			SortKey: compute.SortKey{ColumnIndex: path[0], Order: compute.SortOrderAscending, NullPlacement: compute.SortNullsAtEnd},
		}
		if field.Direction == SortDESC {
			key.Order = compute.SortOrderDescending
		}
		if field.NullOrder == NullsFirst {
			key.NullPlacement = compute.SortNullsAtStart
		}
		keys = append(keys, key)
	}

	return keys, nil
}

func sortFieldPath(fileSchema *iceberg.Schema, fields []iceberg.NestedField, sourceID, orderID int) ([]int, error) {
	if path, ok := structFieldPath(fields, sourceID); ok {
		return path, nil
	}

	if _, exists := fileSchema.FindFieldByID(sourceID); exists {
		return nil, fmt.Errorf("sort order %d: source id %d is nested under a list or map, which is not supported",
			orderID, sourceID)
	}

	return nil, fmt.Errorf("sort order %d: source id %d is not a column in schema",
		orderID, sourceID)
}

// structFieldPath returns the field-index path to sourceID walking only
// structs. Sources under lists or maps are not found.
func structFieldPath(fields []iceberg.NestedField, sourceID int) ([]int, bool) {
	for i, f := range fields {
		if f.ID == sourceID {
			return []int{i}, true
		}
		st, ok := f.Type.(*iceberg.StructType)
		if !ok {
			continue
		}
		inner, ok := structFieldPath(st.FieldList, sourceID)
		if !ok {
			continue
		}

		return append([]int{i}, inner...), true
	}

	return nil, false
}

func allTopLevelSortKeys(keys []sortKey) bool {
	for _, k := range keys {
		if len(k.path) != 1 {
			return false
		}
	}

	return true
}

// applySortKeys returns a sorted copy of batch according to keys. Top-level
// keys use compute.SortRecordBatch; nested keys extract leaf columns (merging
// ancestor struct nulls into the leaf) and sort via indices + take.
func applySortKeys(ctx context.Context, batch arrow.RecordBatch, keys []sortKey) (arrow.RecordBatch, error) {
	if len(keys) == 0 {
		batch.Retain()

		return batch, nil
	}

	if allTopLevelSortKeys(keys) {
		computeKeys := make([]compute.SortKey, len(keys))
		for i, k := range keys {
			computeKeys[i] = k.SortKey
			computeKeys[i].ColumnIndex = k.path[0]
		}

		return compute.SortRecordBatch(ctx, batch, computeKeys)
	}

	return sortRecordBatchByPaths(ctx, batch, keys)
}

func sortRecordBatchByPaths(ctx context.Context, batch arrow.RecordBatch, keys []sortKey) (arrow.RecordBatch, error) {
	mem := compute.GetAllocator(ctx)
	extracted := make([]arrow.Array, len(keys))
	fields := make([]arrow.Field, len(keys))
	defer func() {
		for _, col := range extracted {
			if col != nil {
				col.Release()
			}
		}
	}()

	for i, k := range keys {
		col, err := extractSortColumn(batch, k.path, mem)
		if err != nil {
			return nil, err
		}
		extracted[i] = col
		fields[i] = arrow.Field{
			Name:     fmt.Sprintf("_sort_key_%d", i),
			Type:     col.DataType(),
			Nullable: true,
		}
	}

	sortBatch := array.NewRecordBatch(arrow.NewSchema(fields, nil), extracted, batch.NumRows())
	defer sortBatch.Release()

	computeKeys := make([]compute.SortKey, len(keys))
	for i, k := range keys {
		computeKeys[i] = compute.SortKey{
			ColumnIndex: i, Order: k.Order, NullPlacement: k.NullPlacement,
		}
	}

	indices, err := compute.SortIndices(ctx, compute.NewDatumWithoutOwning(sortBatch), compute.SortOptions(computeKeys))
	if err != nil {
		return nil, err
	}
	defer indices.Release()

	resultDatum, err := compute.Take(ctx, *compute.DefaultTakeOptions(),
		compute.NewDatumWithoutOwning(batch), indices)
	if err != nil {
		return nil, err
	}

	result := resultDatum.(*compute.RecordDatum).Value
	result.Retain()
	resultDatum.Release()

	return result, nil
}

func extractSortColumn(batch arrow.RecordBatch, path []int, mem memory.Allocator) (arrow.Array, error) {
	chain, err := resolveColumnChain(batch, path)
	if err != nil {
		return nil, err
	}

	leaf := chain[len(chain)-1]
	if len(chain) == 1 {
		leaf.Retain()

		return leaf, nil
	}

	return withAncestorNulls(leaf, chain[:len(chain)-1], mem)
}

// withAncestorNulls returns a view of leaf whose validity is the intersection
// of leaf and every ancestor struct. A null parent therefore sorts as a null
// key, even when the child storage still holds a value.
func withAncestorNulls(leaf arrow.Array, ancestors []arrow.Array, mem memory.Allocator) (arrow.Array, error) {
	needMerge := false
	for _, a := range ancestors {
		if a.NullN() != 0 {
			needMerge = true

			break
		}
	}
	if !needMerge {
		leaf.Retain()

		return leaf, nil
	}

	n := leaf.Len()
	offset := leaf.Data().Offset()
	raw := mem.Allocate(int(bitutil.BytesForBits(int64(offset + n))))
	for i := range raw {
		raw[i] = 0
	}
	nullCount := 0
	for i := range n {
		valid := !leaf.IsNull(i)
		if valid {
			for _, a := range ancestors {
				if a.IsNull(i) {
					valid = false

					break
				}
			}
		}
		if valid {
			bitutil.SetBit(raw, offset+i)
		} else {
			nullCount++
		}
	}

	validity := memory.NewBufferWithAllocator(raw, mem)
	// buffers[0] is the validity bitmap for every layout except unions, which
	// have none at index 0; Iceberg has no union type, so this always holds.
	buffers := slices.Clone(leaf.Data().Buffers())
	if len(buffers) == 0 {
		buffers = []*memory.Buffer{validity}
	} else {
		buffers[0] = validity
	}
	data := array.NewData(leaf.DataType(), n, buffers, leaf.Data().Children(), nullCount, offset)
	validity.Release()
	out := array.MakeFromData(data)
	data.Release()

	return out, nil
}
