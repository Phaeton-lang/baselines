#!/bin/bash
#python gbdt.py --lms 224 5
python gbdt.py --lms 36 5 2>&1 | tee lms-gbdt-36.log
python gbdt.py --lms 48 5 2>&1 | tee lms-gbdt-48.log
python gbdt.py --lms 56 5 2>&1 | tee lms-gbdt-56.log
python gbdt.py --lms 64 5 2>&1 | tee lms-gbdt-64.log
python gbdt.py --lms 96 5 2>&1 | tee lms-gbdt-96.log
