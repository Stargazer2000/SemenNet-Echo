import numpy

# nums=[[[3,3,8],[9,4,2],[5,3,4]],
#       [[5,5,4],[4,3,6],[7,8,5]],
#       [[3,2,6],[5,7,8],[7,4,2]]
#       ]
nums=[[3,3,8],[9,4,2],[5,3,4]]
print("nums[0]:",nums[0])
print("nums[1]:",nums[1])
print("nums[2]:",nums[2])
mask = (nums >= 0) & (nums <10)
print(mask)