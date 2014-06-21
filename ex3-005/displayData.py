"""
%DISPLAYDATA Display 2D data in a nice grid
%   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
%   stored in X in a nice grid. It returns the figure handle h and the 
%   displayed array if requested.


"""
def displayData(X):
    #import numpy for matrix manipuations
    import numpy as np
    #setting colormap 
    import matplotlib.pyplot as plt

    import matplotlib.image as mpimg

    #set example width if not provided

    if 'example_width' in locals():
        example_width=example_width.astype(int)
    else:
        example_width=np.round(np.sqrt(np.size(X,1))).astype(int)
    

    #Gray Image

    

    # Compute rows, cols
    import pdb
    pdb.set_trace()
    m,n = np.shape(X) 
    example_height = (n / example_width);

    #compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)));
    display_cols= int(np.ceil(m / display_rows));

    # Between images padding
    pad=1

    display_array = - np.ones([(pad + display_rows * (example_height + pad)), ( pad + display_cols * (example_width + pad))])

    #% Copy each example into a patch on the display array
    curr_ex=1
    for j in range (1,display_rows+1):
        for i in range(1,display_cols+1):
            if curr_ex > m:
                break
            # Copy the patch
                    
            # Get the max value of the patch
            max_val = np.max(abs(X[curr_ex, :]))
            print(max_val)
            display_array[pad +(j-1) *(example_height+pad) + (1,example_height), pad +(i-1)* (example_width+pad)+(1,example_width)]=\
                    X[curr_ex,:].reshape(example_height,example_width)#/max_val
            curr_ex=curr_ex+1
        if curr_ex > m:
            break

    imgplot = plt.imshow(display_array)
    return curr_ex
