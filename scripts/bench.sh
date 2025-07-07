#!/bin/sh
RUSTFLAGS="-C target-cpu=native" cargo bench --package thunder --benches --all-features
MEAN_NANOS=$(jq '.mean | .point_estimate' target/criterion/connect_blocks/new/estimates.json)
MEDIAN_NANOS=$(jq '.median | .point_estimate' target/criterion/connect_blocks/new/estimates.json)
MEAN_S=$(echo "scale=2; $MEAN_NANOS / 1000000000" | bc -l)
MEDIAN_S=$(echo "scale=2; $MEDIAN_NANOS / 1000000000" | bc -l)
echo "*************"
echo "YOUR SCORE:  ${MEAN_S}"
echo "Verified 5x 160mb blocks in ${MEAN_S} seconds (mean) / ${MEDIAN_S} seconds (median)."
echo "*************"