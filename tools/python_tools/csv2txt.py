import csv

infile = csv.reader(open('/home/fyx/lab404_ws/src/SC-LIO-SAM/Mulran_DCC01.csv', 'r'))

# write
with open('/home/fyx/lab404_ws/src/SC-LIO-SAM/Mulran_DCC01_kitti.txt', 'w') as f1:
	while infile:
		for line in infile:
			# print(line)
			i = 1
			for id in line:
				f1.write(id)
				if i != 12 :
					f1.write(' ')
					i = i + 1		
			f1.write('\n')
f1.close()

print("done")
