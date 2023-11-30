cimport cython
import numpy as np
cimport numpy as np

cdef find_new_cells(list point, long[:, :] fd_array):

    cdef list new_cells

    cdef int row, col

    new_cells = []
    row, col = point[0], point[1]
    if fd_array[row, col + 1] == 5:
        new_cells.append([row, col + 1])
    if fd_array[row - 1, col + 1] == 6:
        new_cells.append([row - 1, col + 1])
    if fd_array[row - 1, col] == 7:
        new_cells.append([row - 1, col])
    if fd_array[row - 1, col - 1] == 8:
        new_cells.append([row - 1, col - 1])
    if fd_array[row, col - 1] == 1:
        new_cells.append([row, col - 1])
    if fd_array[row + 1, col - 1] == 2:
        new_cells.append([row + 1, col - 1])
    if fd_array[row + 1, col] == 3:
        new_cells.append([row + 1, col])
    if fd_array[row + 1, col + 1] == 4:
        new_cells.append([row + 1, col + 1])

    return new_cells


@cython.boundscheck(False)
@cython.wraparound(False)
def upstream_basin(list pour_point, long[:, :] fd_array):

    cdef Py_ssize_t x_max = fd_array.shape[0]
    cdef Py_ssize_t y_max = fd_array.shape[1]

    out_array = np.zeros((x_max, y_max), dtype=np.intc)
    cdef int[:, :] out_view = out_array

    cdef int row, col

    cdef list cell, nc

    row, col = pour_point[0], pour_point[1]
    # set the pour point as part of the basin
    out_view[row, col] = 1

    new_cells = find_new_cells(pour_point, fd_array)
    while len(new_cells) > 0:
        for cell in new_cells:
            if out_view[cell[0], cell[1]] != 1:
                out_view[cell[0], cell[1]] = 1
                new_cells2 = find_new_cells(cell, fd_array)
                new_cells.remove(cell)
                for nc in new_cells2:
                    new_cells.append(nc)

    return out_array