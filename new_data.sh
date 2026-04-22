#!/bin/bash

set -e

rm -rf data/raw/camera/front_left
rm -rf data/raw/camera/front_right
rm -rf data/raw/lidar/front_top
rm -rf data/raw/radar/front_right
rm -rf data/raw/radar/front_left

invoke read-mcap
