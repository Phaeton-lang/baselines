#!/bin/bash
python kmeans.py --no-lms 256 256 10 2>&1 | tee nolms-kmeans-256-256.log
python kmeans.py --lms 36 256 5 2>&1 | tee lms-kmeans-36-256.log
python kmeans.py --lms 48 256 5 2>&1 | tee lms-kmeans-48-256.log
python kmeans.py --lms 64 256 5 2>&1 | tee lms-kmeans-64-256.log
python kmeans.py --lms 128 256 5 2>&1 | tee lms-kmeans-128-256.log
python kmeans.py --lms 256 256 5 2>&1 | tee lms-kmeans-256-256.log
cd ../decision-tree/
./py-run.sh
./lms-run.sh
