#mulran_data to kitti

from pyparsing import line

data_set = []

gt_file = '/home/fyx/lab404_ws/src/SC-LIO-SAM/Mulran_DCC01.txt'
gt_kitti_file = '/home/fyx/lab404_ws/src/SC-LIO-SAM/Mulran_DCC01_kitti.txt'

# read
with open(gt_file, "r") as f:
    line = f.readline()
    while line:
	line = line.strip('\n')
	line = line.strip('\r')
        data = (line.split(','))
	print(data)
        data_set.append(data)
f.close()

# print(data_set)

# write
with open(gt_kitti_file, 'w') as f1:
    for data_line in data_set:
        for i in range(2, len(data_line)):
            f1.write(data_line[i])
            print(i)
        f1.write('\n')
f1.close()
