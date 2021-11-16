# vim:fenc=utf-8
# vim:ft=pyrex
import cupy


include "cupy_const_extrap.pyx"
include "cupy_linear_extrap.pyx"


_const_extrap_factory = {
    ("west", 0): _const_extrap_west_w,
    ("west", 1): _const_extrap_west_hu,
    ("west", 2): _const_extrap_west_hv,
    ("east", 0): _const_extrap_east_w,
    ("east", 1): _const_extrap_east_hu,
    ("east", 2): _const_extrap_east_hv,
    ("south", 0): _const_extrap_south_w,
    ("south", 1): _const_extrap_south_hu,
    ("south", 2): _const_extrap_south_hv,
    ("north", 0): _const_extrap_north_w,
    ("north", 1): _const_extrap_north_hu,
    ("north", 2): _const_extrap_north_hv,
}


_linear_extrap_factory = {
    ("west", 0): _linear_extrap_west_w,
    ("west", 1): _linear_extrap_west_hu,
    ("west", 2): _linear_extrap_west_hv,
    ("east", 0): _linear_extrap_east_w,
    ("east", 1): _linear_extrap_east_hu,
    ("east", 2): _linear_extrap_east_hv,
    ("south", 0): _linear_extrap_south_w,
    ("south", 1): _linear_extrap_south_hu,
    ("south", 2): _linear_extrap_south_hv,
    ("north", 0): _linear_extrap_north_w,
    ("north", 1): _linear_extrap_north_hu,
    ("north", 2): _linear_extrap_north_hv,
}
