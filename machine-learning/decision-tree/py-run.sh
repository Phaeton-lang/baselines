#!/bin/bash
#python gbdt.py --no-lms 32 5 2>&1 | tee nolms-gbdt-32.log
python gbdt.py --no-lms 36 5 2>&1 | tee nolms-gbdt-36.log
python gbdt.py --no-lms 48 5 2>&1 | tee nolms-gbdt-48.log
python gbdt.py --no-lms 56 5 2>&1 | tee nolms-gbdt-56.log
python gbdt.py --no-lms 64 5 2>&1 | tee nolms-gbdt-64.log
python gbdt.py --no-lms 96 5 2>&1 | tee nolms-gbdt-96.log
