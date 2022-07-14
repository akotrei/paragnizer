import numpy as np
import ctypes

W = 2048 
H = 2896

COLS = [
    'page_n',
    'word',
    'x1',
    'y1',
    'x2',
    'y2',
    'bold',
    'par_ord',
    'par_bel',
]

RE_PUNKT = r'(^\s*\d+\.\d*\.?)|(^\s*\(?[a-zA-Z]\))|(^\s*\(?\d+\))|([livxIVX]+[.)])|([a-zA-Z]\.)|(\([livxIVX]+\))|(\[?\d+/\d+\]?)'


def get_hfd_lib(path):
    """ Returns header_footer_detect.so lib and any function can be invoked from it
    """
    lib = ctypes.pydll.LoadLibrary(path)
    lib.head_foot_det.restype = None
    return lib

def sort_by_lines(lib_hfd, y1, y2, line_id, sorted=True):
    """ Sorts words by lines in @line_id
        @y1, @y2, @line_id - numpy arrays with shape (N,), dtype = int32,
        where N - number of words. @y1, @y2 - top and down positions of the words
        @line_id should contain only zeros!!!
        if @sorted is False this function sort y1 and y2 so that is needed
    """
    if sorted is False:
        y1_index = y1.argsort()
        y1 = y1[y1_index]
        y2 = y2[y1_index]

    lib_hfd.head_foot_det(ctypes.py_object(y1),
                          ctypes.py_object(y2),
                          ctypes.py_object(line_id))
    return line_id

def get_features():
    features = {
    'x1': np.float32(0),
    'y1': np.float32(0),
    'x2': np.float32(0),
    'y2': np.float32(0),
    'bold': np.float32(0),
    'punkt': np.float32(0),
    }
    return features


def get_classes():
    classes = {
        'forward': np.float32(0),
        'line': np.float32(0),
        'forward_p': np.float32(0),
        'line_p': np.float32(0),
    }
    return classes

def fill_moves(df_page):
    result = []

    df_pars = [x for _, x in df_page.groupby('par_bel')]
    df_pars = [df_par.sort_values('par_ord') for df_par in df_pars]

    pars = len(df_pars) - 1
    for i, df_par in enumerate(df_pars):
        forward = (df_par['x1'].diff(-1) < 0).astype('int32')
        line = (df_par['x1'].diff(-1) > 0).astype('int32')

        df_par['forward'] = forward
        df_par['line'] = line
        df_par['forward_p'] = 0
        df_par['line_p'] = 0

        if i < pars:
            df_par_next = df_pars[i+1]

            # next paragraph is right from current
            if df_par_next['x1'].min() >= df_par['x2'].max():
                df_par.loc[df_par.index[-1], 'forward_p'] = 1
            # next paragraph is bellow
            else:
                df_par.loc[df_par.index[-1], 'line_p'] = 1
        # next paragraph is bellow
        else:
            df_par.loc[df_par.index[-1], 'line_p'] = 1
        
        result.append(df_par)

    return pd.concat(result)



if __name__ == '__main__':
    csv_exp_path = 'dataset/01/output.csv'
    # csv_exp_path = 'dataset/Fieldglass Framework Subcontracor Agreement Draft Template v2_2 Clean version 16102020/output.csv'

    import pandas as pd

    df = pd.read_csv(csv_exp_path)
    df = df[COLS]
    df['punkt'] = df['word'].str.match(RE_PUNKT).astype('int32')
    df = df.reindex(columns = df.columns.tolist() + list(get_classes().keys()) + ['line_id'])

    df_page = df[df['page_n']==1]

    df_page = fill_moves(df_page)

    

    lib = get_hfd_lib('paragnizer/generator/libhfd.so')

    ind = df_page.sort_values('y1').index

    y1 = df.loc[ind, 'y1'].to_numpy().astype('int32')
    y2 = df.loc[ind, 'y2'].to_numpy().astype('int32')

    line_id = np.zeros_like(y1)
    line_id = sort_by_lines(lib, y1, y2, line_id)

    df_page.loc[ind, 'line_id'] = line_id

    print(df_page[list(get_classes().keys())].sum(axis=1).unique())
    print(df_page.to_string())




