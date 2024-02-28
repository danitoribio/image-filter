
import numpy as np

# In[ ]:
import multiprocessing as mp
from multiprocessing.sharedctypes import Value, Array, RawArray
from multiprocessing import Process, Lock
import ctypes
import myFunctions as my


# Exercise 1 function

def image_filter(image, filter_mask, numprocessors, filtered_image):

        
        rows=range(image.shape[0])
        filter_mask = my.filter(image,filter_mask) # As we are using the filter class we need to create a filter object

        # We create a pool to filter in parallel all the rows of the image
        with mp.Pool(processes=numprocessors,initializer=my.pool_init1,initargs=[filtered_image,image,filter_mask]) as p:
            p.map(filter_mask.edge_filter1,rows)

        
                
        return filtered_image

def image_filter2(image, filter_mask, numprocessors, filtered_image):

        
        rows=range(image.shape[0])
        filter_mask = my.filter(image,filter_mask) # As we are using the filter class we need to create a filter object

        # We create a pool to filter in parallel all the rows of the image
        with mp.Pool(processes=numprocessors,initializer=my.pool_init2,initargs=[filtered_image,image,filter_mask]) as p:
            p.map(filter_mask.edge_filter2,rows)

        
                
        return filtered_image

# Exercise 2

def filters_execution(image, filter_mask1, filter_mask2, numprocessors, filtered_image1, filtered_image2):
    # We define the two processes to be executed at the same time
    first_process = mp.Process(target = image_filter, args = (image, filter_mask1, numprocessors, filtered_image1))
    second_process = mp.Process(target = image_filter2, args = (image, filter_mask2, numprocessors, filtered_image2))
  
    # Each process starts to run
    first_process.start()
    second_process.start()
    
    # We wait until both processes are finished
    first_process.join()
    second_process.join()



    return

