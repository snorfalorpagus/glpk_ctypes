#!/usr/bin/env python

import os
import sys
from ctypes import *
from ctypes.util import find_library

# TODO: sensible defaults for different operating systems
if sys.platform == 'windows':
    is_64bits = sys.maxsize > 2**32
    if is_64bits:
        lib_name = 'glpk_4_55_w64.dll'
    else:
        lib_name = 'glpk_4_55_w32.dll'

    lib_name = os.path.join(os.path.dirname(__file__), lib_name)
elif sys.platform == 'darwin':
    lib_name = find_library('glpk')

lib = CDLL(lib_name)

DBL_MAX = sys.float_info.max

# optimization direction flag
GLP_MIN = 1 # minimization
GLP_MAX = 2 # maximization

# kind of structural variable
GLP_CV = 1 # continuous variable
GLP_IV = 2 # integer variable
GLP_BV = 3 # binary variable

# type of auxiliary/structural variable
GLP_FR = 1 # free (unbounded) variable
GLP_LO = 2 # variable with lower bound
GLP_UP = 3 # variable with upper bound
GLP_DB = 4 # double-bounded variable
GLP_FX = 5 # fixed variable

GLP_MSG_OFF = 0 # no output
GLP_MSG_ERR = 1 # warning and error messages only
GLP_MSG_ON = 2 # normal output
GLP_MSG_ALL = 3 # full output
GLP_MSG_DBG = 4 # debug output

# enable/disable flag
GLP_ON = 1 # enable something
GLP_OFF = 0 # disable something

# solution status
# includes short description from pyglpk
solution_status = {
    1: ('GLP_UNDEF', 'solution is undefined', 'undef'),
    2: ('GLP_FEAS', 'solution is feasible', 'feas'),
    3: ('GLP_INFEAS', 'solution is infeasible', 'infeas'),
    4: ('GLP_NOFEAS', 'no feasible solution exists', 'nofeas'),
    5: ('GLP_OPT', 'solution is optimal', 'opt'),
    6: ('GLP_UNBND', 'solution is unbounded', 'unbnd'),
}
for value, (key, ldescription, sdescription) in solution_status.items():
    locals()[key] = value

# return codes
return_codes = {
    0x01: ('GLP_EBADB', 'invalid basis',),
    0x02: ('GLP_ESING', 'singular matrix',),
    0x03: ('GLP_ECOND', 'ill-conditioned matrix',),
    0x04: ('GLP_EBOUND', 'invalid bounds',),
    0x05: ('GLP_EFAIL', 'solver failed',),
    0x06: ('GLP_EOBJLL', 'objective lower limit reached',),
    0x07: ('GLP_EOBJUL', 'objective upper limit reached',),
    0x08: ('GLP_EITLIM', 'iteration limit exceeded',),
    0x09: ('GLP_ETMLIM', 'time limit exceeded',),
    0x0A: ('GLP_ENOPFS', 'no primal feasible solution',),
    0x0B: ('GLP_ENODFS', 'no dual feasible solution',),
    0x0C: ('GLP_EROOT', 'root LP optimum not provided',),
    0x0D: ('GLP_ESTOP', 'search terminated by application',),
    0x0E: ('GLP_EMIPGAP', 'relative mip gap tolerance reached',),
    0x0F: ('GLP_ENOFEAS', 'no primal/dual feasible solution',),
    0x10: ('GLP_ENOCVG', 'no convergence',),
    0x11: ('GLP_EINSTAB', 'numerical instability',),
    0x12: ('GLP_EDATA', 'invalid data',),
    0x13: ('GLP_ERANGE', 'result out of range',),
}
for value, (key, description) in return_codes.items():
    locals()[key] = value

glp_prob_p = c_void_p

class glp_smcp(Structure):
    _fields_ = [
        ('msg_lev', c_int),
        ('meth', c_int),
        ('pricing', c_int),
        ('r_test', c_int),
        ('tol_bnd', c_double),
        ('tol_dj', c_double),
        ('tol_piv', c_double),
        ('obj_ll', c_double),
        ('obj_ul', c_double),
        ('it_lim', c_int),
        ('tm_lim', c_int),
        ('out_frq', c_int),
        ('out_dly', c_int),
        ('presolve', c_int),
        ('foo_bar', c_double*36),
    ]

lib.glp_create_prob.argtypes = None
lib.glp_create_prob.restype = glp_prob_p
lib.glp_set_prob_name.argtypes = [glp_prob_p, c_char_p]
lib.glp_set_prob_name.restype = None
lib.glp_set_obj_name.argtypes = [glp_prob_p, c_char_p]
lib.glp_set_obj_name.restype = None
lib.glp_set_obj_dir.argtypes = [glp_prob_p, c_int]
lib.glp_set_obj_dir.restype = None
lib.glp_add_rows.argtypes = [glp_prob_p, c_int]
lib.glp_add_rows.restype = c_int
lib.glp_add_cols.argtypes = [glp_prob_p, c_int]
lib.glp_add_cols.restype = c_int
lib.glp_set_row_name.argtypes = [glp_prob_p, c_int, c_char_p]
lib.glp_set_row_name.restype = None
lib.glp_set_col_name.argtypes = [glp_prob_p, c_int, c_char_p]
lib.glp_set_col_name.restype = None
lib.glp_set_row_bnds.argtypes = [glp_prob_p, c_int, c_int, c_double, c_double]
lib.glp_set_row_bnds.restype = None
lib.glp_set_col_bnds.argtypes = [glp_prob_p, c_int, c_int, c_double, c_double]
lib.glp_set_col_bnds.restype = None
lib.glp_set_obj_coef.argtypes = [glp_prob_p, c_int, c_double]
lib.glp_set_obj_coef.restype = None
lib.glp_set_mat_col.argtypes = [glp_prob_p, c_int, c_int, c_void_p, c_void_p]
lib.glp_set_mat_col.restype = None
lib.glp_set_mat_row.argtypes = [glp_prob_p, c_int, c_int, c_void_p, c_void_p]
lib.glp_set_mat_row.restype = None
lib.glp_load_matrix.argtypes = [glp_prob_p, c_int, POINTER(c_int), POINTER(c_int), POINTER(c_double)]
lib.glp_load_matrix.restype = None
lib.glp_del_rows.argtypes = [glp_prob_p, c_int, POINTER(c_int)]
lib.glp_del_rows.restype = None
lib.glp_del_cols.argtypes = [glp_prob_p, c_int, POINTER(c_int)]
lib.glp_del_cols.restype = None
lib.glp_delete_prob.argtypes = [glp_prob_p]
lib.glp_delete_prob.restype = None

lib.glp_get_prob_name.argtypes = [glp_prob_p]
lib.glp_get_prob_name.restype = c_char_p
lib.glp_get_obj_name.argtypes = [glp_prob_p]
lib.glp_get_obj_name.restype = c_char_p
lib.glp_get_obj_dir.argtypes = [glp_prob_p]
lib.glp_get_obj_dir.restype = c_int
lib.glp_get_num_rows.argtypes = [glp_prob_p]
lib.glp_get_num_rows.restype = c_int
lib.glp_get_num_cols.argtypes = [glp_prob_p]
lib.glp_get_num_cols.restype = c_int
lib.glp_get_row_name.argtypes = [glp_prob_p, c_int]
lib.glp_get_row_name.restype = c_char_p
lib.glp_get_col_name.argtypes = [glp_prob_p, c_int]
lib.glp_get_col_name.restype = c_char_p
lib.glp_get_row_lb.argtypes = [glp_prob_p, c_int]
lib.glp_get_row_lb.restype = c_double
lib.glp_get_row_ub.argtypes = [glp_prob_p, c_int]
lib.glp_get_row_ub.restype = c_double
lib.glp_get_col_lb.argtypes = [glp_prob_p, c_int]
lib.glp_get_col_lb.restype = c_double
lib.glp_get_col_ub.argtypes = [glp_prob_p, c_int]
lib.glp_get_col_ub.restype = c_double
lib.glp_get_obj_coef.argtypes = [glp_prob_p, c_int]
lib.glp_get_obj_coef.restype = c_double
lib.glp_get_num_nz.argtypes = [glp_prob_p]
lib.glp_get_num_nz.restype = c_int
lib.glp_get_mat_row.argtypes = [glp_prob_p, c_int, POINTER(c_int), POINTER(c_double)]
lib.glp_get_mat_row.restype = c_int
lib.glp_get_mat_col.argtypes = [glp_prob_p, c_int, POINTER(c_int), POINTER(c_double)]
lib.glp_get_mat_col.restype = c_int
lib.glp_get_obj_val.argtypes = [glp_prob_p]
lib.glp_get_obj_val.restype = c_double
lib.glp_get_row_prim.argtypes = [glp_prob_p, c_int]
lib.glp_get_row_prim.restype = c_double
lib.glp_get_row_dual.argtypes = [glp_prob_p, c_int]
lib.glp_get_row_dual.restype = c_double
lib.glp_get_col_prim.argtypes = [glp_prob_p, c_int]
lib.glp_get_col_prim.restype = c_double
lib.glp_get_col_dual.argtypes = [glp_prob_p, c_int]
lib.glp_get_col_dual.restype = c_double

lib.glp_create_index.argtypes = [glp_prob_p]
lib.glp_create_index.restype = None
lib.glp_find_row.argtypes = [glp_prob_p, c_char_p]
lib.glp_find_row.restype = c_int
lib.glp_find_col.argtypes = [glp_prob_p, c_char_p]
lib.glp_find_col.restype = c_int

lib.glp_simplex.argtypes = [glp_prob_p, POINTER(glp_smcp)]
lib.glp_simplex.restype = c_int
lib.glp_init_smcp.argtypes = [POINTER(glp_smcp)]
lib.glp_init_smcp.restype = c_int
lib.glp_get_status.argtypes = [glp_prob_p]
lib.glp_get_status.restype = c_int

lib.glp_set_col_kind.argtypes = [glp_prob_p, c_int, c_int]
lib.glp_set_col_kind.restype = None
lib.glp_get_col_kind.argtypes = [glp_prob_p, c_int]
lib.glp_get_col_kind.restype = c_int
lib.glp_get_num_int.argtypes = [glp_prob_p]
lib.glp_get_num_int.restype = c_int
lib.glp_get_num_bin.argtypes = [glp_prob_p]
lib.glp_get_num_bin.restype = c_int

lib.glp_set_rii.argtypes = [glp_prob_p, c_int, c_double]
lib.glp_set_rii.restype = None
lib.glp_set_sjj.argtypes = [glp_prob_p, c_int, c_double]
lib.glp_set_sjj.restype = None
lib.glp_get_rii.argtypes = [glp_prob_p, c_int]
lib.glp_get_rii.restype = c_double
lib.glp_get_sjj.argtypes = [glp_prob_p, c_int]
lib.glp_get_sjj.restype = c_double
lib.glp_scale_prob.argtypes = [glp_prob_p, c_int]
lib.glp_scale_prob.restype = None
lib.glp_unscale_prob.argtypes = [glp_prob_p]
lib.glp_unscale_prob.restype = None

lib.glp_version.argtypes = None
lib.glp_version.restype = c_char_p
lib.glp_term_out.argtypes = [c_int]
lib.glp_term_out.restype = c_int
lib.glp_mem_usage.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_size_t), POINTER(c_size_t)]
lib.glp_mem_usage.restype = None
lib.glp_mem_limit.argtypes = [c_int]
lib.glp_mem_limit.restype = None

kinds = {'c': 'col', 'r': 'row'}

class Environment(object):
    def __init__(self):
        self._mem_limit = None
    
    @property
    def version(self):
        return tuple([int(i) for i in lib.glp_version().split('.')])
    
    def term_on(self, onoff):
        if onoff:
            lib.glp_term_out(GLP_ON)
        else:
            lib.glp_term_out(GLP_OFF)
    term_on = property(lambda self: None, term_on)
    
    def term_hook(self, func):
        if func is None:
            lib.glp_term_hook(None, None)
        else:
            # need to keep reference to things, as ctypes doesn't
            TERM_FUNC = self._TERM_FUNC = CFUNCTYPE(c_int, c_void_p, c_char_p)
            def wrapper(info, line):
                ret = func(line)
                if ret is not None:
                    return int(ret)
                else:
                    return 0 # TODO: check this is correct
            term_func = self._term_func = TERM_FUNC(wrapper)
            lib.glp_term_hook(term_func, None)
    term_hook = property(lambda self: None, term_hook)
    
    def _mem_usage(self):
        count = c_int()
        cpeak = c_int()
        total = c_size_t()
        tpeak = c_size_t()
        lib.glp_mem_usage(byref(count), byref(cpeak), byref(total), byref(tpeak))
        return count.value, cpeak.value, total.value, tpeak.value
    
    @property
    def blocks(self):
        return int(self._mem_usage()[0])
    
    @property
    def blocks_peak(self):
        return int(self._mem_usage()[1])
    
    @property
    def bytes(self):
        return int(self._mem_usage()[2])

    @property
    def bytes_peak(self):
        return int(self._mem_usage()[3])
    
    @property
    def mem_limit(self):
        return self._mem_limit
    
    @mem_limit.setter
    def mem_limit(self, max_megabytes):
        '''Set the memory limit
        
        GLPK doesn't provide a function to get the current memory limit, so
        we have to do it ourselves.
        '''
        if max_megabytes is None or max_megabytes == 0:
            self._mem_limit = max_megabytes
            max_megabytes = 0x7fffffff
        else:
            if not isinstance(max_megabytes, int):
                raise TypeError()
            if max_megabytes < 0:
                raise ValueError()
            self._mem_limit = max_megabytes
            max_megabytes = c_int(max_megabytes)
        lib.glp_mem_limit(max_megabytes)
    
    def __delattr__(self, name):
        if name == 'mem_limit':
            self.mem_limit = None
        elif name == 'term_hook':
            self.term_hook = None
        else:
            raise AttributeError(name)

env = Environment()

class LP(object):
    def __init__(self, gmp=None, mps=None, freemps=None, cpxlp=None, glp=None):
        self._lp = lib.glp_create_prob()
        self.cols = BarCollection('c', self)
        self.rows = BarCollection('r', self)
        self.obj = Objective(self)
        
        # TODO: read gmp, mps, lp, etc.
    
    MSG_OFF = GLP_MSG_OFF
    MSG_ERR = GLP_MSG_ERR
    MSG_ON = GLP_MSG_ON
    MSG_ALL = GLP_MSG_ALL
    MSG_DBG = GLP_MSG_DBG
    
    @property
    def kind(self):
        if lib.glp_get_num_int(self._lp) > 0:
            return int
        else:
            return float
    
    @property
    def name(self):
        return lib.glp_get_prob_name(self._lp)
    @name.setter
    def name(self, name):
        if name is not None:
            if not isinstance(name, basestring):
                raise TypeError()
            name = name.decode('ascii')
            assert(len(name) <= 255)
        return lib.glp_set_prob_name(self._lp, name)
    
    @property
    def matrix(self):
        # here we are lazy, depending on Bar.matrix
        retval = []
        for row in self.rows:
            rowmat = row.matrix
            for (col_num, val) in rowmat:
                retval.append((row.index, col_num, val))
        return retval
    
    @matrix.setter
    def matrix(self, mat):
        if not isinstance(mat, (tuple, list,)):
            raise TypeError()
        matrix = {}
        for n, item in enumerate(mat):
            i, j, value = item
            i = self.rows[i]._index
            j = self.rows[j]._index
            index = (i, j,)
            if index in matrix:
                raise ValueError('duplicate index {},{} detected'.format(i,j))
            matrix[index] = value
        ne = len(matrix)
        ia = (c_int*(ne+1))()
        ja = (c_int*(ne+1))()
        ar = (c_double*(ne+1))()
        for n, ((i, j), value) in enumerate(matrix.items()):
            ia[n+1] = i+1
            ja[n+1] = j+1
            ar[n+1] = value
        lib.glp_load_matrix(self._lp, ne, ia, ja, ar)
    
    @property
    def nnz(self):
        return lib.glp_get_num_nz(self._lp)
    
    def simplex(self, **kwargs):
        cp = glp_smcp()
        lib.glp_init_smcp(cp)
        cp.msg_lev = GLP_MSG_OFF
        for key, value in kwargs.items():
            if hasattr(cp, key):
                if key in ('it_lim', 'msg_lev', 'out_dly', 'tm_lim'):
                    if value < 0:
                        raise ValueError()
                if key in ('out_frq', 'tol_dj', 'tol_piv', 'tol_bnd'):
                    if value <= 0:
                        raise ValueError()
                if key in ('tol_dj', 'tol_piv', 'tol_bnd'):
                    if value >= 1: # TODO FIXME: check the actual conditions for this
                        raise ValueError()
                setattr(cp, key, value)
            else:
                raise TypeError("'{}' is an invalid keyword argument for this function".format(key))
        retval = lib.glp_simplex(self._lp, cp)
        return retval
    
    # TODO: interior, et al
    
    @property
    def status(self):
        status_code = lib.glp_get_status(self._lp)
        return solution_status[status_code][2]
    
    # TODO: lp.matrix (getter+setter)
    
    # TODO: lp.write()
    
    # TODO: lp.scale(), lp.unscale()
    
    def __delattr__(self, name):
        if name == 'name':
            self.name = None
        else:
            raise AttributeError()
    
    def __del__(self):
        lib.glp_delete_prob(self._lp)
    
    def __repr__(self):
        return '<LP {rows}-by-{cols} at {hex}>'.format(rows=len(self.rows), cols=len(self.cols), hex=hex(id(self)))

# for compatibility with pyglpk
LPX = LP

class BarCollection(object):
    def __init__(self, kind, parent):
        self._kind = kind
        self._parent = parent
        self._lp = parent._lp
    
    def add(self, count):
        if not isinstance(count, int):
            raise TypeError()
        if count < 1:
            raise ValueError()
        if self._kind == 'c':
            return lib.glp_add_cols(self._lp, count) - 1
        else:
            return lib.glp_add_rows(self._lp, count) - 1
    
    def __len__(self):
        if self._kind == 'c':
            return lib.glp_get_num_cols(self._lp)
        else:
            return lib.glp_get_num_rows(self._lp)
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            bars = []
            for index in range(*key.indices(len(self))):
                bars.append(Bar(self._kind, self, index))
            return bars
        elif isinstance(key, (tuple, list)):
            bars = []
            for index in key:
                if not isinstance(index, (int, basestring)):
                    raise TypeError()
                bar = self[index]
                bars.append(bar)
            return bars
        elif isinstance(key, int):
            if key < 0:
                key = len(self)+key
            if key < 0 or key >= len(self):
                raise IndexError() #raise TypeError()
            return Bar(self._kind, self, key)
        elif isinstance(key, basestring):
            # find bar by name
            lib.glp_create_index(self._lp)
            if self._kind == 'c':
                ind = lib.glp_find_col(self._lp, key)
            else:
                ind = lib.glp_find_row(self._lp, key)
            if ind == 0:
                raise KeyError()
            else:
                return Bar(self._kind, self, ind-1)
        else:
            raise TypeError()
    
    def __iter__(self):
        for n in range(len(self)):
            yield self[n]

    def __delitem__(self, key):
        bars = self[key]
        if isinstance(bars, Bar):
            inds = [bars._index+1]
        else:
            if len(bars) == 0:
                return
            inds = [b._index+1 for b in bars]
        num = (c_int*(len(inds)+1))()
        num[1:] = inds
        if self._kind == 'c':
            lib.glp_del_cols(self._lp, len(inds), num)
        else:
            lib.glp_del_rows(self._lp, len(inds), num)
    
    def __contains__(self, name):
        lib.glp_create_index(self._lp)
        if self._kind == 'c':
            ind = lib.glp_find_col(self._lp, name)
        else:
            ind = lib.glp_find_row(self._lp, name)
        if ind > 0:
            return True
        else:
            return False
    
    def __repr__(self):
        return '<BarCollection {kind} at {hex}>'.format(kind=kinds[self._kind]+'s', hex=hex(id(self)))

def is_valid(f):
    '''Check if a bar's index is still valid'''
    def wrapper(self, *args, **kwargs):
        if not self._index < len(self._parent):
            raise RuntimeError('row or column no longer valid')
        return f(self, *args, **kwargs)
    return wrapper

class Bar(object):
    def __init__(self, kind, parent, index):
        self._kind = kind
        self._parent = parent
        self._lp = self._parent._parent._lp
        self._index = index

    @property
    def index(self):
        return self._index
    
    @property
    @is_valid
    def name(self):
        if self._kind == 'c':
            return lib.glp_get_col_name(self._lp, self._index+1)
        else:
            return lib.glp_get_row_name(self._lp, self._index+1)
    @name.setter
    @is_valid
    def name(self, name):
        if name is not None:
            if not isinstance(name, basestring):
                raise TypeError()
            name = name.decode('ascii')
            assert(len(name) <= 255)
        if self._kind == 'c':
            lib.glp_set_col_name(self._lp, self._index+1, name)
        else:
            lib.glp_set_row_name(self._lp, self._index+1, name)
    
    # TODO: del(lp.row[0].name)
    
    @property
    @is_valid
    def bounds(self):
        if self._kind == 'c':
            lb = lib.glp_get_col_lb(self._lp, self._index+1)
            ub = lib.glp_get_col_ub(self._lp, self._index+1)
        else:
            lb = lib.glp_get_row_lb(self._lp, self._index+1)
            ub = lib.glp_get_row_ub(self._lp, self._index+1)
        if lb == -DBL_MAX:
            lb = None
        if ub == DBL_MAX:
            ub = None
        return lb, ub
    @bounds.setter
    @is_valid
    def bounds(self, bounds):
        if isinstance(bounds, (int, float,)) or bounds is None:
            lb = bounds
            ub = bounds
        elif isinstance(bounds, (tuple, list,)):
            if len(bounds) == 2:
                lb, ub = bounds
            else:
                raise TypeError() # TODO: check exception type
        else:
            raise TypeError()
        if not isinstance(lb, (int, float)) and lb is not None:
            raise TypeError()
        if not isinstance(ub, (int, float)) and ub is not None:
            raise TypeError()
        if lb is None:
            lb = -DBL_MAX
        if ub is None:
            ub = DBL_MAX
        if lb == ub:
            bound_type = GLP_FX
        elif lb is None:
            if ub is None:
                bound_type = GLP_FR
            else:
                bound_type = GLP_UP
        elif ub is None:
            bound_type = GLP_LO
        else:
            if lb > ub:
                raise ValueError()
            bound_type = GLP_DB
        if self._kind == 'c':
            lib.glp_set_col_bnds(self._lp, self._index+1, bound_type, lb, ub)
        else:
            lib.glp_set_row_bnds(self._lp, self._index+1, bound_type, lb, ub)
    
    @property
    @is_valid
    def matrix(self):
        if self._kind == 'c':
            if len(self._parent._parent.rows) == 0:
                return []
            get_mat = lib.glp_get_mat_col
        else:
            if len(self._parent._parent.cols) == 0:
                return []
            get_mat = lib.glp_get_mat_row
        length = get_mat(self._lp, self._index+1, None, None)
        if length == 0:
            return []
        ind = (c_int*(length+1))()
        val = (c_double*(length+1))()
        length = get_mat(self._lp, self._index+1, ind, val)
        mat = []
        if length == 0:
            return mat
        for n in range(0, length):
            mat.append((ind[n+1]-1, val[n+1]))
        return mat
    
    @matrix.setter
    @is_valid
    def matrix(self, mat):
        if self._kind == 'c':
            set_mat = lib.glp_set_mat_col
            bar_len = len(self._parent._parent.rows)
        else:
            set_mat = lib.glp_set_mat_row
            bar_len = len(self._parent._parent.cols)
        mat_len = len(mat)
        ind = (c_int*(mat_len+1))()
        val = (c_double*(mat_len+1))()
        if isinstance(mat[0], (tuple, list,)):
            for n, (i, v) in enumerate(mat):
                if i < 0:
                    raise IndexError()
                if i >= bar_len: # CHECK THIS
                    raise IndexError()
                ind[n+1] = i+1
                val[n+1] = v
        elif isinstance(mat[0], (float, int,)):
            for n in range(0, len(mat)):
                if n >= bar_len:
                    raise IndexError()
                ind[n+1] = n+1
                val[n+1] = mat[n]
        set_mat(self._lp, self._index+1, mat_len, byref(ind), byref(val))
    
    @property
    @is_valid
    def kind(self):
        if self._kind == 'c':
            k = lib.glp_get_col_kind(self._lp, self._index+1)
            if k == GLP_CV:
                return float
            elif k == GLP_IV:
                return int
            else:
                return bool
        else:
            return float
    
    @kind.setter
    @is_valid
    def kind(self, kind):
        if self._kind == 'c':
            if kind is float:
                kind = GLP_CV
            elif kind is int:
                kind = GLP_IV
            elif kind is bool:
                kind = GLP_BV
            else:
                raise ValueError('either the type float, int, or bool is required')
            lib.glp_set_col_kind(self._lp, self._index+1, kind)
        else:
            if kind is float:
                pass
            elif kind is int:
                raise ValueError('row variables cannot be integer')
            elif kind is bool:
                raise ValueError('row variables cannot be binary')
            else:
                raise ValueError('either the type float, int, or bool is required')
    
    @property
    def iscol(self):
        return self._kind == 'c'
    
    @property
    def isrow(self):
        return self._kind == 'r'
    
    @property
    @is_valid
    def scale(self):
        if self._kind == 'c':
            return lib.glp_get_sjj(self._lp, self._index+1)
        else:
            return lib.glp_get_rii(self._lp, self._index+1)
    
    @scale.setter
    @is_valid
    def scale(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError()
        if value <= 0:
            raise ValueError()
        if self._kind == 'c':
            lib.glp_set_sjj(self._lp, self._index+1, value)
        else:
            lib.glp_set_rii(self._lp, self._index+1, value)
    
    # TODO: status, one of 'bs', 'nl','nu','nf','ns'
    
    @property
    @is_valid
    def primal(self):
        val = lib.glp_get_row_prim(self._lp, self._index+1)
        return val
    
    @property
    @is_valid
    def dual(self):
        val = lib.glp_get_row_dual(self._lp, self._index+1)
        return val
    
    def __eq__(self, x):
        if not isinstance(x, Bar):
            return False # not a bar
        elif self._parent != x._parent:
            return False # different parents
        elif self._kind != x._kind:
            return False # different kind
        elif self._index != x._index:
            return False # different index
        else:
            return True
    
    def __gt__(self, x):
        if not isinstance(x, Bar):
            return True
        elif self._parent != x._parent:
            return True
        elif self._index > x._index:
            return True
        else:
            return False
    
    def __delattr__(self, name):
        if name == 'name':
            self.name = None
        elif name == 'bounds':
            self.bounds = None
        else:
            raise AttributeError()
    
    # TODO: scale, lp.rows[i].scale = factor
        
    def __repr__(self):
        return '<Bar {kind} {index} at {hex}>'.format(kind=kinds[self._kind], index=self._index, hex=hex(id(self)))

class Objective(object):
    def __init__(self, parent):
        self._parent = parent
        self._lp = parent._lp
    
    @property
    def name(self):
        return lib.glp_get_obj_name(self._lp)
    
    @name.setter
    def name(self, name):
        if name is not None:
            if not isinstance(name, basestring):
                raise TypeError()
            name = name.decode('ascii')
            assert(len(name) <= 255)
        lib.glp_set_obj_name(self._lp, name)

    @property
    def minimize(self):
        return lib.glp_get_obj_dir(self._lp) == GLP_MIN
    @minimize.setter
    def minimize(self, value):
        if value:
            value = GLP_MIN
        else:
            value = GLP_MAX
        lib.glp_set_obj_dir(self._lp, value)
    
    @property
    def maximize(self):
        return not self.minimize
    @maximize.setter
    def maximize(self, value):
        self.minimize = not value
    
    @property
    def value(self):
        return lib.glp_get_obj_val(self._lp)
    
    def __len__(self):
        return len(self._parent.cols)
    
    def __getitem__(self, key):
        if key is None:
            return lib.glp_get_obj_coef(self._lp, 0)
        cols = self._parent.cols[key]
        if isinstance(cols, Bar):
            return lib.glp_get_obj_coef(self._lp, cols.index+1)
        else:
            ret = []
            for col in cols:
                ret.append(lib.glp_get_obj_coef(self._lp, col.index+1))
            return ret
    
    def __iter__(self):
        for n in range(len(self)):
            yield self[n]
    
    def __setitem__(self, key, value):
        if key is None:
            if not isinstance(value, (float, int)):
                raise TypeError()
            lib.glp_set_obj_coef(self._lp, 0, value)
            return
        cols = self._parent.cols[key]
        if isinstance(cols, Bar):
            cols = [cols]
        if isinstance(value, (float, int)):
            for col in cols:
                lib.glp_set_obj_coef(self._lp, col.index+1, value)
        elif isinstance(value, (list, tuple)):
            if len(cols) == 1:
                raise TypeError()
            if len(value) == len(cols):
                for col, val in zip(cols, value):
                    if not isinstance(val, (float, int)):
                        raise TypeError()
                    lib.glp_set_obj_coef(self._lp, col.index+1, val)
            else:
                raise ValueError()
        else:
            raise TypeError()
    
    @property
    def shift(self):
        return self[None]
    @shift.setter
    def shift(self, value):
        self[None] = value
    
    def __delattr__(self, name):
        if name == 'name':
            self.name = None
        else:
            raise AttributeError()
    
    def __repr__(self):
        return '<Objective at {hex}>'.format(hex=hex(id(self)))
