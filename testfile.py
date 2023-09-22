# trying to parse a weights file
from collections.abc import Iterable

def parser(d_st, lvl, inx):
    tp = type(d_st)
    print("Type = {}, index = {},level={}".format(tp,inx,lvl))
    if isinstance(d_st, Iterable):
        i = 1
        for ele in d_st:
            parser(ele, lvl + 1,i)
            i+=1
            if i>10:
                print("Terminating level {} at current index".format(lvl+1))
                break

parser([1,[2,4,6,range(25),8,9],3,4],1,1)