# 实验五错误案例人工归因模板

> 下面案例由数据自动抽取；请人工复核 attribution 字段。

## Case 1: misclassification · pandas-dev/pandas#66061 · C1_P1

- 自动归因建议：局部 diff 信号弱 / AI 代码风格统一导致模型过度乐观或保守（需人工复核）
- 人工归因（待填）：
- 证据摘录：

```diff
--- pandas/io/parsers/arrow_parser_wrapper.py ---
@@ -54,6 +54,10 @@ def _parse_kwds(self) -> None:
             raise ValueError(
                 "The pyarrow engine doesn't support passing a dict for na_values"
             )
+        # pyarrow's null_values only accepts strings, so reject any non-string
+        #  na_values up front instead of silently ignoring them.
+        if not all(isinstance(na_value, str) for na_value in na_values):
+            raise TypeError("The 'pyarrow' engine requires all na_values to be strings")
         self.na_values = list(self.kwds["na_values"])
 
     def _get_pyarrow_options(self) -> None:
@@ -152,19 +156,13 @@ def _get_convert_options(self):
 
         try:
             convert_options = pyarrow_csv.ConvertOptions(**self.convert_options)
-        except TypeError as err:
+        except TypeError:
+            # Non-string na_values are rejected in _parse_kwds, so any remaining
+            #  TypeError here is from invalid usecols/include_columns.
             include = self.convert_options.get("include_columns", None)
             if include is not None:
                 self._validate_usecols(include)
 
-            nulls = self.conve
```

## Case 2: misclassification · pandas-dev/pandas#66042 · C1_P1

- 自动归因建议：局部 diff 信号弱 / AI 代码风格统一导致模型过度乐观或保守（需人工复核）
- 人工归因（待填）：
- 证据摘录：

```diff
--- pandas/core/dtypes/missing.py ---
@@ -532,7 +532,12 @@ def _array_equivalent_object(
         left_remaining = left
         right_remaining = right
 
-    for left_value, right_value in zip(left_remaining, right_remaining, strict=True):
+    # left_remaining/right_remaining may be 2-D here (e.g. DataFrame block
+    # values that reached this fallback); use .flat to compare element-wise
+    # rather than row-wise (GH#43008)
+    for left_value, right_value in zip(
+        left_remaining.flat, right_remaining.flat, strict=True
+    ):
         if left_value is NaT and right_value is not NaT:
             return False
 
@@ -542,6 +547,12 @@ def _array_equivalent_object(
         elif isinstance(left_value, float) and np.isnan(left_value):
             if not isinstance(right_value, float) or not np.isnan(right_value):
                 return False
+        elif isinstance(left_value, (ABCSeries, ABCDataFrame)):
+            # GH#43008 nested Series/DataFrame: recurse via equals so that NaNs
+            # in the same location compare equal (a plain `!=` treats NaN as
+            # unequal to NaN). equals also returns False for a type mismatch.
+            if not left_value.e
```

## Case 3: misclassification · pandas-dev/pandas#65940 · C1_P1

- 自动归因建议：局部 diff 信号弱 / AI 代码风格统一导致模型过度乐观或保守（需人工复核）
- 人工归因（待填）：
- 证据摘录：

```diff
--- pandas/tests/scalar/timedelta/test_constructors.py ---
@@ -656,6 +656,10 @@ def test_construction_out_of_bounds_td64s(val, unit):
         ("PT-6H3M", Timedelta(hours=-6, minutes=3)),
         ("-PT6H3M", Timedelta(hours=-6, minutes=-3)),
         ("-PT-6H+3M", Timedelta(hours=6, minutes=-3)),
+        # GH#48122: hour/minute components may have more than two digits
+        ("PT100H", Timedelta(hours=100)),
+        ("PT100M", Timedelta(minutes=100)),
+        ("P0DT999H999M999S", Timedelta(hours=999, minutes=999, seconds=999)),
     ],
 )
 def test_iso_constructor(fmt, exp):
@@ -667,7 +671,6 @@ def test_iso_constructor(fmt, exp):
     [
         "PPPPPPPPPPPP",
         "PDTHMS",
-        "P0DT999H999M999S",
         "P1DT0H0M0.0000000000000S",
         "P1DT0H0M0.S",
         "P",
```

## Case 4: misclassification · pandas-dev/pandas#65951 · C1_P1

- 自动归因建议：局部 diff 信号弱 / AI 代码风格统一导致模型过度乐观或保守（需人工复核）
- 人工归因（待填）：
- 证据摘录：

```diff
--- pandas/tests/io/parser/common/test_float.py ---
@@ -140,6 +140,18 @@ def test_precise_xstrtod_large_mantissa(c_parser_only, value):
     assert result == float(value)
 
 
+def test_precise_xstrtod_leading_zeros(c_parser_only):
+    # GH#64184
+    # Leading zeros must not consume the 17-significant-digit budget in the
+    # precise_xstrtod fallback (reached when a thousands separator is set),
+    # which would otherwise push trailing significant digits into the
+    # exponent, e.g. "000000000010084566" -> 10084560.0 instead of 10084566.0.
+    parser = c_parser_only
+    data = "val\n000000000010084566.0\n"
+    result = parser.read_csv(StringIO(data), thousands=",")["val"][0]
+    assert result == 10084566.0
+
+
 @pytest.mark.parametrize(
     "value", ["81e31d04049863b72", "d81e31d04049863b72", "81e3104049863b72"]
 )
--- pandas/tests/tools/test_to_numeric.py ---
@@ -910,3 +910,29 @@ def test_large_exponent_coerce():
     result = to_numeric(ser, errors="coerce")
     expected = Series([np.inf])
     tm.assert_series_equal(result, expected)
+
+
+@pytest.mark.parametrize(
+    "data, expected",
+    [
+        # mixing in a float forces the float64 result dtype
+        ([10
```

## Case 5: misclassification · pandas-dev/pandas#65976 · C1_P1

- 自动归因建议：局部 diff 信号弱 / AI 代码风格统一导致模型过度乐观或保守（需人工复核）
- 人工归因（待填）：
- 证据摘录：

```diff
--- pandas/core/computation/ops.py ---
@@ -445,25 +445,32 @@ def stringify(value):
                 encoder = pprint_thing
             return encoder(value)
 
+        def convert(value):
+            if isinstance(value, (int, float)):
+                value = stringify(value)
+            value = Timestamp(ensure_decoded(value))
+            if value.tz is not None:
+                value = value.tz_convert("UTC")
+            return value
+
+        def convert_term(term) -> None:
+            value = term.value
+            if term.is_scalar:
+                term.update(convert(value))
+            elif isinstance(value, list):
+                # ``==``/``in`` comparisons are rewritten to membership ops
+                # with the right-hand side wrapped in a list (see
+                # _rewrite_membership_op), so convert the elements too
+                # (GH#35595)
+                term.update([convert(element) for element in value])
+
         lhs, rhs = self.lhs, self.rhs
 
-        if is_term(lhs) and lhs.is_datetime and is_term(rhs) and rhs.is_scalar:
-            v = rhs.value
-            if isinstance(v, (int, float)):
-                v = stringify(v)
-           
```
