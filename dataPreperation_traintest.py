import os
import numpy as np

script_dir = os.path.dirname(__file__)

# Declaring paths for Training Set
fine_ape = "dataset/fine/ape"
fine_ape_quaternion = "dataset/fine/ape/poses.txt"

fine_benchvise = "dataset/fine/benchvise"
fine_benchvise_quaternion = "dataset/fine/benchvise/poses.txt"

fine_cam = "dataset/fine/cam"
fine_cam_quaternion = "dataset/fine/cam/poses.txt"

fine_cat = "dataset/fine/cat"
fine_cat_quaternion = "dataset/fine/cat/poses.txt"

fine_duck = "dataset/fine/duck"
fine_duck_quaternion = "dataset/fine/duck/poses.txt"

# Declaring paths for training and test set
real_ape = "dataset/real/ape"
real_ape_quaternion = "dataset/real/ape/poses.txt"

real_benchvise = "dataset/real/benchvise"
real_benchvise_quaternion = "dataset/real/benchvise/poses.txt"

real_cam = "dataset/real/cam"
real_cam_quaternion = "dataset/real/cam/poses.txt"

real_cat = "dataset/real/cat"
real_cat_quaternion = "dataset/real/cat/poses.txt"

real_duck = "dataset/real/duck"
real_duck_quaternion = "dataset/real/duck/poses.txt"
training_split_file = "dataset/real/training_split.txt"

# Creating the Training Set
training_split = open(os.path.join(script_dir,training_split_file), 'r').read().split(", ")
training_split = [int(data) for data in training_split[0:len(training_split)-1]]
print(np.shape(training_split)[0])

training_set_image_paths = []
training_set_quaternions = []
test_set_image_paths = []
test_set_quaternions = []

# Appending the real images for ape into the training and test sets
text = open(os.path.join(script_dir, real_ape_quaternion), 'r').read().split('\n')
real_ape_qs = [data for data in text if not data.__contains__("#")]
real_ape_qs = real_ape_qs[0:len(real_ape_qs)-1]
real_ape_qs = [data.split(" ") for data in real_ape_qs]
for i in range(np.shape(real_ape_qs)[0]):
    filename = "real" + str(i) + ".png"
    if training_split.__contains__(i):
        training_set_image_paths.append(os.path.join(real_ape,filename))
        training_set_quaternions.append(real_ape_qs[i])
    else:
        test_set_image_paths.append(os.path.join(real_ape,filename))
        test_set_quaternions.append(real_ape_qs[i])
# Appending the fine images for ape into training set
text = open(os.path.join(script_dir, fine_ape_quaternion), 'r').read().split('\n')
fine_ape_qs = [data for data in text if not data.__contains__("#")]
fine_ape_qs = fine_ape_qs[0:len(fine_ape_qs)-1]
fine_ape_qs = [data.split(" ") for data in fine_ape_qs]
for i in range(np.shape(fine_ape_qs)[0]):
    filename = "fine" + str(i) + ".png"
    training_set_image_paths.append(os.path.join(fine_ape,filename))
    training_set_quaternions.append(fine_ape_qs[i])
# print(np.shape(training_set_image_paths))
# print(np.shape(training_set_quaternions))
# print(np.shape(test_set_image_paths))
# print(np.shape(test_set_quaternions))

# Appending the real images for benchvise into the training and test sets
text = open(os.path.join(script_dir, real_benchvise_quaternion), 'r').read().split('\n')
real_benchvise_qs = [data for data in text if not data.__contains__("#")]
real_benchvise_qs = real_benchvise_qs[0:len(real_benchvise_qs)-1]
real_benchvise_qs = [data.split(" ") for data in real_benchvise_qs]
for i in range(np.shape(real_benchvise_qs)[0]):
    filename = "real" + str(i) + ".png"
    if training_split.__contains__(i):
        training_set_image_paths.append(os.path.join(real_benchvise,filename))
        training_set_quaternions.append(real_benchvise_qs[i])
    else:
        test_set_image_paths.append(os.path.join(real_benchvise,filename))
        test_set_quaternions.append(real_benchvise_qs[i])
# Appending the fine images for benchvise into training set
text = open(os.path.join(script_dir, fine_benchvise_quaternion), 'r').read().split('\n')
fine_benchvise_qs = [data for data in text if not data.__contains__("#")]
fine_benchvise_qs = fine_benchvise_qs[0:len(fine_benchvise_qs)-1]
fine_benchvise_qs = [data.split(" ") for data in fine_benchvise_qs]
for i in range(np.shape(fine_benchvise_qs)[0]):
    filename = "fine" + str(i) + ".png"
    training_set_image_paths.append(os.path.join(fine_benchvise,filename))
    training_set_quaternions.append(fine_benchvise_qs[i])
# print(np.shape(training_set_image_paths))
# print(np.shape(training_set_quaternions))
# print(np.shape(test_set_image_paths))
# print(np.shape(test_set_quaternions))

# Appending the real images for cam into the training and test sets
text = open(os.path.join(script_dir, real_cam_quaternion), 'r').read().split('\n')
real_cam_qs = [data for data in text if not data.__contains__("#")]
real_cam_qs = real_cam_qs[0:len(real_cam_qs)-1]
real_cam_qs = [data.split(" ") for data in real_cam_qs]
for i in range(np.shape(real_cam_qs)[0]):
    filename = "real" + str(i) + ".png"
    if training_split.__contains__(i):
        training_set_image_paths.append(os.path.join(real_cam,filename))
        training_set_quaternions.append(real_cam_qs[i])
    else:
        test_set_image_paths.append(os.path.join(real_cam,filename))
        test_set_quaternions.append(real_cam_qs[i])
# Appending the fine images for cam into training set
text = open(os.path.join(script_dir, fine_cam_quaternion), 'r').read().split('\n')
fine_cam_qs = [data for data in text if not data.__contains__("#")]
fine_cam_qs = fine_cam_qs[0:len(fine_cam_qs)-1]
fine_cam_qs = [data.split(" ") for data in fine_cam_qs]
for i in range(np.shape(fine_cam_qs)[0]):
    filename = "fine" + str(i) + ".png"
    training_set_image_paths.append(os.path.join(fine_cam,filename))
    training_set_quaternions.append(fine_cam_qs[i])
# print(np.shape(training_set_image_paths))
# print(np.shape(training_set_quaternions))
# print(np.shape(test_set_image_paths))
# print(np.shape(test_set_quaternions))

# Appending the real images for cat into the training and test sets
text = open(os.path.join(script_dir, real_cat_quaternion), 'r').read().split('\n')
real_cat_qs = [data for data in text if not data.__contains__("#")]
real_cat_qs = real_cat_qs[0:len(real_cat_qs)-1]
real_cat_qs = [data.split(" ") for data in real_cat_qs]
for i in range(np.shape(real_cat_qs)[0]):
    filename = "real" + str(i) + ".png"
    if training_split.__contains__(i):
        training_set_image_paths.append(os.path.join(real_cat,filename))
        training_set_quaternions.append(real_cat_qs[i])
    else:
        test_set_image_paths.append(os.path.join(real_cat,filename))
        test_set_quaternions.append(real_cat_qs[i])
# Appending the fine images for cat into training set
text = open(os.path.join(script_dir, fine_cat_quaternion), 'r').read().split('\n')
fine_cat_qs = [data for data in text if not data.__contains__("#")]
fine_cat_qs = fine_cat_qs[0:len(fine_cat_qs)-1]
fine_cat_qs = [data.split(" ") for data in fine_cat_qs]
for i in range(np.shape(fine_cat_qs)[0]):
    filename = "fine" + str(i) + ".png"
    training_set_image_paths.append(os.path.join(fine_cat,filename))
    training_set_quaternions.append(fine_cat_qs[i])
# print(np.shape(training_set_image_paths))
# print(np.shape(training_set_quaternions))
# print(np.shape(test_set_image_paths))
# print(np.shape(test_set_quaternions))

# Appending the real images for duck into the training and test sets
text = open(os.path.join(script_dir, real_duck_quaternion), 'r').read().split('\n')
real_duck_qs = [data for data in text if not data.__contains__("#")]
real_duck_qs = real_duck_qs[0:len(real_duck_qs)-1]
real_duck_qs = [data.split(" ") for data in real_duck_qs]
print(np.shape(real_duck_qs))
for i in range(np.shape(real_duck_qs)[0]):
    filename = "real" + str(i) + ".png"
    if training_split.__contains__(i):
        training_set_image_paths.append(os.path.join(real_duck,filename))
        training_set_quaternions.append(real_duck_qs[i])
    else:
        test_set_image_paths.append(os.path.join(real_duck,filename))
        test_set_quaternions.append(real_duck_qs[i])
# Appending the fine images for duck into training set
text = open(os.path.join(script_dir, fine_duck_quaternion), 'r').read().split('\n')
fine_duck_qs = [data for data in text if not data.__contains__("#")]
fine_duck_qs = fine_duck_qs[0:len(fine_duck_qs)-1]
fine_duck_qs = [data.split(" ") for data in fine_duck_qs]
for i in range(np.shape(fine_duck_qs)[0]):
    filename = "fine" + str(i) + ".png"
    training_set_image_paths.append(os.path.join(fine_duck,filename))
    training_set_quaternions.append(fine_duck_qs[i])
# print(np.shape(training_set_image_paths))
# print(np.shape(training_set_quaternions))
# print(np.shape(test_set_image_paths))
# print(np.shape(test_set_quaternions))

# Write the train.txt and test.txt files
writeFile = open("train.txt",'w')
for i in range(np.shape(training_set_image_paths)[0]):
    writeFile.write(training_set_image_paths[i])
    writeFile.write(";")
    for j in range(4):
        writeFile.write(training_set_quaternions[i][j])
        if j is not 3:
            writeFile.write(";")
        else:
            writeFile.write("\n")

writeFile = open("test.txt",'w')
for i in range(np.shape(test_set_image_paths)[0]):
    writeFile.write(test_set_image_paths[i])
    writeFile.write(";")
    for j in range(4):
        writeFile.write(test_set_quaternions[i][j])
        if j is not 3:
            writeFile.write(";")
        else:
            writeFile.write("\n")


exit(0)

