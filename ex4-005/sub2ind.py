

def sub2ind(shape, row_sub, col_sub):
        """
        Return the linear index equivalents to the row and column subscripts for given matrix shape.

        :param shape: Preferred matrix shape for subscripts conversion.
        :type shape: `tuple`
        :param row_sub: Row subscripts.
        :type row_sub: `list`
        :param col_sub: Column subscripts.
        :type col_sub: `list`
        """
        m=shape[0];
        assert len(row_sub) == len(col_sub), "Row and column subscripts do not match."
        res = [j * 10 + i for i, j in zip( row_sub,col_sub)]
        return res
        

