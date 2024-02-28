
#Needed libraries

import numpy as np

# In[ ]:
import multiprocessing as mp
from multiprocessing.sharedctypes import Value, Array, RawArray
from multiprocessing import Process, Lock
import ctypes


# In[ ]:
#This functions just create a numpy array structure of type unsigned int8, with the memory used by our global r/w shared memory
def tonumpyarray(mp_arr):
    #mp_array is a shared memory array with lock
    
    return np.frombuffer(mp_arr.get_obj(),dtype=np.uint8)
    

def pool_init1(shared_array_,srcimg, imgfilter):
    
    """
    This function defines the global variables that will be used in the 
    filtering process.
      
    INPUTS
    shared_array_ : space where the threads will store the results
    srcimg --> Array: image to be filtered
    imgfilter --> Array: the filter we will apply to the image
    
    """

    # All the global variables that will be used in the filtering process are initialized here:
    global shared_space1  # --> reference space where the matrix with the results are stored
    global shared_matrix1 # --> matrix where all the threads will write the filtered value of a pixel
    global image_1 	      # --> the image that to be filtered
    global my_filter1     # --> the filter that is applied to the image

    # All the global variables that will be used in the filtering process are defined here:
    image_1 = srcimg
    my_filter1 = imgfilter
    size = image_1.shape
    shared_space1 = shared_array_
    shared_matrix1 = tonumpyarray(shared_space1).reshape(size)


# We need another pool for the second process
def pool_init2(shared_array_,srcimg, imgfilter):
    
    """
    This function defines the global variables that will be used in the 
    filtering process.
      
    INPUTS
    shared_array_ : space where the threads will store the results
    srcimg --> Array: image to be filtered
    imgfilter --> Array: the filter we will apply to the image
    
    """

    # All the global variables that will be used in the filtering process are initialized here:
    global shared_space2 # --> reference space where the matrix with the results are stored
    global shared_matrix2 # --> matrix where all the threads will write the filtered value of a pixel
    global image_2	      # --> the image that to be filtered
    global my_filter2     # --> the filter that is applied to the image

    # All the global variables that will be used in the filtering process are defined here:
    image_2 = srcimg
    my_filter2 = imgfilter
    size = image_2.shape
    shared_space2 = shared_array_
    shared_matrix2 = tonumpyarray(shared_space2).reshape(size)

#This function initialize the global shared memory data
class filter:
    
    def __init__(self,srcimg, imgfilter):
        #shared_array_: is the shared read/write data, with lock. It is a vector (because the shared memory should be allocated as a vector
        #srcimg: is the original image
        #imgfilter is the filter which will be applied to the image and stor the results in the shared memory array
        
        #We defines the local process memory reference for shared memory space
        #Assign the shared memory  to the local reference
        
        #Here, we will define the read only memory data as global (the scope of this global variables is the local module)
        
        self.image = srcimg
        self.my_filter = imgfilter
        
        #here, we initialize the global read only memory data
        self.size = srcimg.shape
        
        
        #Defines the numpy matrix reference to handle data, which will uses the shared memory buffer

# In[ ]:     
#this function just copy the original image to the global r/w shared  memory 
    def parallel_shared_imagecopy(self,row):

        global shared_space
        global shared_matrix
        
        image = self.image
        my_filter = self.my_filter
        # with this instruction we lock the shared memory space, avoiding other parallel processes tries to write on it
        with shared_space.get_lock():
            #while we are in this code block no ones, except this execution thread, can write in the shared memory
            shared_matrix[row,:,:]=image[row,:,:]
        return

# In[ ]:

# With this function we filter a row and write it in the shared space
    def edge_filter1(self,row):

        # We first call all the global variables to be used  (defined in the Pool_init 1 function)
        global shared_space1
        global shared_matrix1
        
        image = self.image
        my_filter = self.my_filter
        
        (rows,cols,depth) = image.shape
        (filter_rows,filter_cols)=my_filter.shape
        #fetch the r row from the original image
        srow=image[row,:,:]

        # here we obtain the previous row of srow
        if ( row>0 ): # The row is not close to the upper border. We can use the previous previous row
            prow=image[row-1,:,:]
        else: # The row is close to the upper border. We cannot use the previous previous row
            prow=image[row,:,:]
        
        # here we have to obtain the previous row of prow
        if row>1: # The row is not close to the upper border. We can use the previous previous row
            prevprow = image[row-2,:,:]
        else: # The row is close to the upper border. We cannot use the previous previous row
            prevprow = image[row, :, :] 
        
        # here we obtain the next row of srow
        if ( row == (rows-1)):  # The row is close to the lower border. We cannot use the next row
            nrow=image[row,:,:]
        else: # The row is not close to the lower border. We can use the next row
            nrow=image[row+1,:,:]

        # here we obtain the next row of nrow
        if ( row >= (rows-2)):  # The row is close to the lower border. We cannot use the next next row
            nextnrow=nrow
        else:  # The row is not close to the lower border. We can use the next next row
            nextnrow=image[row+2,:,:]
    
        #defines the result vector, and set the initial value to 0
        frow=np.zeros_like(srow)
    
        for d in range(depth):
            for idx in range(cols):

                mat = np.zeros((5,5)) # we define the matrix which will store the values of the near pixels, we start with the 
                                            # most general case and then take only the pixels we need depending on the filter shape
                
            
                # Here we obtain the central column of the pixels matrix
                mat[0,2] = prevprow[idx,d]
                mat[1,2] = prow[idx,d]
                mat[2,2] = srow[idx,d]
                mat[3,2] = nrow[idx,d]
                mat[4,2] = nextnrow[idx,d]
                

                # Here we obtain the pixels in the previous column
                if idx == 0:  ### pixel in first column of original image matrix
                    mat[0,1] = prevprow[idx,d]
                    mat[1,1] = prow[idx,d]
                    mat[2,1] = srow[idx,d]
                    mat[3,1] = nrow[idx,d]
                    mat[4,1] = nextnrow[idx,d]

                else:
                    mat[0,1] = prevprow[idx-1,d]
                    mat[1,1] = prow[idx-1,d]
                    mat[2,1] = srow[idx-1,d]
                    mat[3,1] = nrow[idx-1,d]
                    mat[4,1] = nextnrow[idx-1,d]

                # Here we obtain the pixels in the first column
                if idx<1: ### pixel in the first two columns of the original image matrix
                    mat[0,0] = prevprow[idx,d]
                    mat[1,0] = prow[idx,d]
                    mat[2,0] = srow[idx,d]
                    mat[3,0] = nrow[idx,d]
                    mat[4,0] = nextnrow[idx,d]
                else:
                    mat[0,0] = prevprow[idx-2,d]
                    mat[1,0] = prow[idx-2,d]
                    mat[2,0] = srow[idx-2,d]
                    mat[3,0] = nrow[idx-2,d]
                    mat[4,0] = nextnrow[idx-2,d]

                # Here we obtain the pixels in the next column 
                if (idx==(cols-1)):  ### pixel in last column of original image matrix
                    mat[0,3] = prevprow[idx,d]
                    mat[1,3] = prow[idx,d]
                    mat[2,3] = srow[idx,d]
                    mat[3,3] = nrow[idx,d]
                    mat[4,3] = nextnrow[idx,d]
                else:
                    mat[0,3] = prevprow[idx+1,d]
                    mat[1,3] = prow[idx+1,d]
                    mat[2,3] = srow[idx+1,d]
                    mat[3,3] = nrow[idx+1,d]
                    mat[4,3] = nextnrow[idx+1,d]

                # Here we obtain the pixels in the last column
                if (idx>=(cols-2)):  ### pixel in last two columns of original image matrix
                    mat[0,4] = prevprow[idx,d]
                    mat[1,4] = prow[idx,d]
                    mat[2,4] = srow[idx,d]
                    mat[3,4] = nrow[idx,d]
                    mat[4,4] = nextnrow[idx,d]
                else:
                    mat[0,4] = prevprow[idx+2,d]
                    mat[1,4] = prow[idx+2,d]
                    mat[2,4] = srow[idx+2,d]
                    mat[3,4] = nrow[idx+2,d]
                    mat[4,4] = nextnrow[idx+2,d]

            
                # Once we have obtained all the pixels for the matrix we have to take care of the filter shape
                
                
                x = 0
                y = 0 # Case when the filter is 5x5
                if filter_rows == 3:
                    x = 1 # Case when the filter is 3x...
                if filter_cols == 3:
                    y = 1 # Case when the filter is ...x3 
                if filter_rows == 1:
                    x = 2 # Case when the filter is 1x...
                if filter_cols == 1:
                    y = 2 # Case when the filter is ...x1
                
                accu = 0
                for row_ in range(filter_rows):
                    for col_ in range(filter_cols):
                        
                        # We just multiply those pixels inside the filter size region
                        # For example, if the filter is 3x3, the element matrix[0][0] is not used in the              
                        # multiplication, since x and y are 1
                        accu += mat[row_+x][col_+y]*my_filter[row_][col_]   
                        
                frow[idx, d] = accu
                # the pixel's value will correspond with
                # the multiplication of the portion of the image by the filter
                
                
        with shared_space1.get_lock():
            shared_matrix1[row,:,:]=frow # We lock the global variable and modify it
        return frow
    
    

    

# In[ ]:
def edge_filter2(self,row):

        # We first call all the global variables to be used  (defined in the Pool_init 2 function)
        global shared_space2
        global shared_matrix2
        
        image = self.image
        my_filter = self.my_filter
        
        (rows,cols,depth) = image.shape
        (filter_rows,filter_cols)=my_filter.shape
        #fetch the r row from the original image
        srow=image[row,:,:]

        # here we obtain the previous row of srow
        if ( row>0 ): # The row is not close to the upper border. We can use the previous previous row
            prow=image[row-1,:,:]
        else: # The row is close to the upper border. We cannot use the previous previous row
            prow=image[row,:,:]
        
        # here we have to obtain the previous row of prow
        if row>1: # The row is not close to the upper border. We can use the previous previous row
            prevprow = image[row-2,:,:]
        else: # The row is close to the upper border. We cannot use the previous previous row
            prevprow = image[row, :, :] 
        
        # here we obtain the next row of srow
        if ( row == (rows-1)):  # The row is close to the lower border. We cannot use the next row
            nrow=image[row,:,:]
        else: # The row is not close to the lower border. We can use the next row
            nrow=image[row+1,:,:]

        # here we obtain the next row of nrow
        if ( row >= (rows-2)):  # The row is close to the lower border. We cannot use the next next row
            nextnrow=nrow
        else:  # The row is not close to the lower border. We can use the next next row
            nextnrow=image[row+2,:,:]
    
        #defines the result vector, and set the initial value to 0
        frow=np.zeros_like(srow)
    
        for d in range(depth):
            for idx in range(cols):

                mat = np.zeros((5,5)) # we define the matrix which will store the values of the near pixels, we start with the 
                                            # most general case and then take only the pixels we need depending on the filter shape
                
            
                # Here we obtain the central column of the pixels matrix
                mat[0,2] = prevprow[idx,d]
                mat[1,2] = prow[idx,d]
                mat[2,2] = srow[idx,d]
                mat[3,2] = nrow[idx,d]
                mat[4,2] = nextnrow[idx,d]
                

                # Here we obtain the pixels in the previous column
                if idx == 0:  ### pixel in first column of original image matrix
                    mat[0,1] = prevprow[idx,d]
                    mat[1,1] = prow[idx,d]
                    mat[2,1] = srow[idx,d]
                    mat[3,1] = nrow[idx,d]
                    mat[4,1] = nextnrow[idx,d]

                else:
                    mat[0,1] = prevprow[idx-1,d]
                    mat[1,1] = prow[idx-1,d]
                    mat[2,1] = srow[idx-1,d]
                    mat[3,1] = nrow[idx-1,d]
                    mat[4,1] = nextnrow[idx-1,d]

                # Here we obtain the pixels in the first column
                if idx<1: ### pixel in the first two columns of the original image matrix
                    mat[0,0] = prevprow[idx,d]
                    mat[1,0] = prow[idx,d]
                    mat[2,0] = srow[idx,d]
                    mat[3,0] = nrow[idx,d]
                    mat[4,0] = nextnrow[idx,d]
                else:
                    mat[0,0] = prevprow[idx-2,d]
                    mat[1,0] = prow[idx-2,d]
                    mat[2,0] = srow[idx-2,d]
                    mat[3,0] = nrow[idx-2,d]
                    mat[4,0] = nextnrow[idx-2,d]

                # Here we obtain the pixels in the next column 
                if (idx==(cols-1)):  ### pixel in last column of original image matrix
                    mat[0,3] = prevprow[idx,d]
                    mat[1,3] = prow[idx,d]
                    mat[2,3] = srow[idx,d]
                    mat[3,3] = nrow[idx,d]
                    mat[4,3] = nextnrow[idx,d]
                else:
                    mat[0,3] = prevprow[idx+1,d]
                    mat[1,3] = prow[idx+1,d]
                    mat[2,3] = srow[idx+1,d]
                    mat[3,3] = nrow[idx+1,d]
                    mat[4,3] = nextnrow[idx+1,d]

                # Here we obtain the pixels in the last column
                if (idx>=(cols-2)):  ### pixel in last two columns of original image matrix
                    mat[0,4] = prevprow[idx,d]
                    mat[1,4] = prow[idx,d]
                    mat[2,4] = srow[idx,d]
                    mat[3,4] = nrow[idx,d]
                    mat[4,4] = nextnrow[idx,d]
                else:
                    mat[0,4] = prevprow[idx+2,d]
                    mat[1,4] = prow[idx+2,d]
                    mat[2,4] = srow[idx+2,d]
                    mat[3,4] = nrow[idx+2,d]
                    mat[4,4] = nextnrow[idx+2,d]

            
                # Once we have obtained all the pixels for the matrix we have to take care of the filter shape
                
                
                x = 0
                y = 0 # Case when the filter is 5x5
                if filter_rows == 3:
                    x = 1 # Case when the filter is 3x...
                if filter_cols == 3:
                    y = 1 # Case when the filter is ...x3 
                if filter_rows == 1:
                    x = 2 # Case when the filter is 1x...
                if filter_cols == 1:
                    y = 2 # Case when the filter is ...x1
                
                accu = 0
                for row_ in range(filter_rows):
                    for col_ in range(filter_cols):
                        
                        # We just multiply those pixels inside the filter size region
                        # For example, if the filter is 3x3, the element matrix[0][0] is not used in the              
                        # multiplication, since x and y are 1
                        accu += mat[row_+x][col_+y]*my_filter[row_][col_]   
                        
                frow[idx, d] = accu
                # the pixel's value will correspond with
                # the multiplication of the portion of the image by the filter
                
                
        with shared_space2.get_lock():
            shared_matrix2[row,:,:]=frow # We lock the global variable and modify it
        return frow

    


#This cell should be the last one
#this avoid the execution of this script when is invoked directly.
if __name__ == "__main__":
    print("This is not an executable library")



