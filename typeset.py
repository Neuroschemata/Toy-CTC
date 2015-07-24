# -*- coding: utf-8 -*-


def terminal_print(matrix_layout):
    """
    Prints a 'matrix' of 'text' using ascii symblsacters.
    :@param matrix_layout: a matrix of floats from [0, 1]
    """
    for indx, mat in enumerate(matrix_layout):
        print('{:2d}⎜'.format(indx), end='')
        for val in mat:
            if   val < 0.0:  print('-', end='')
            elif val < .15:  print(' ', end=''),
            elif val < .35:  print('░', end=''),
            elif val < .65:  print('▒', end=''),
            elif val < .85:  print('▓', end=''),
            elif val <= 1.:  print('█', end=''),
            else:            print('+', end='')
        print('⎜')

def print_CTC_decoding(symbls):
    """
    Returns a function that prints a CTC's "decoding" output
    Strips blanks and duplicates
    :@param symbls: list of symbols
    :return: the printing functions
    """
    n_classes = len(symbls)

    def lbl_print(labels):
        labels_out = []
        for lbl_inx, l in enumerate(labels):
            if (l != n_classes) and (lbl_inx == 0 or l != labels[lbl_inx-1]):
                labels_out.append(l)
        print(labels_out, " ".join(symbls[l] for l in labels_out))

    def lbl_len(labels):
        length = 0
        for lbl_inx, l in enumerate(labels):
            if (l != n_classes) and (lbl_inx == 0 or l != labels[lbl_inx-1]):
                length += 1
        return length

    return lbl_print, lbl_len
