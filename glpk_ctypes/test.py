#!/usr/bin/env python

import _glpk as glpk

lp = glpk.LPX()
lp.cols.add(3)
lp.rows.add(2)
print(lp)
