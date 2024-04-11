#!/bin/bash
(echo Test 03 && make test3 | grep UNOP && echo Test 02 && make test2 |grep UNOP && echo Test && make test | grep -Ev UNOPTIMIZED | grep OPT) > optimizations.txt && cat optimizations.txt
