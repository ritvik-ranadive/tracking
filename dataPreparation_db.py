import os
import numpy as np

script_dir = os.path.dirname(__file__)

# Declaring file paths for Database Set
coarse_ape = "dataset/coarse/ape"
coarse_ape_quaternion = "dataset/coarse/ape/poses.txt"

coarse_benchvise = "dataset/coarse/benchvise"
coarse_benchvise_quaternion = "dataset/coarse/benchvise/poses.txt"

coarse_cam = "dataset/coarse/cam"
coarse_cam_quaternion = "dataset/coarse/cam/poses.txt"

coarse_cat = "dataset/coarse/cat"
coarse_cat_quaternion = "dataset/coarse/cat/poses.txt"

coarse_duck = "dataset/coarse/duck"
coarse_duck_quaternion = "dataset/coarse/duck/poses.txt"

db_set_image_paths = []
db_set_quaternions = []

# Appending the fine images for ape into db set
text = open(os.path.join(script_dir, coarse_ape_quaternion), 'r').read().split('\n')
coarse_ape_qs = [data for data in text if not data.__contains__("#")]
coarse_ape_qs = coarse_ape_qs[0:len(coarse_ape_qs)-1]
coarse_ape_qs = [data.split(" ") for data in coarse_ape_qs]
for i in range(np.shape(coarse_ape_qs)[0]):
    filename = "coarse" + str(i) + ".png"
    db_set_image_paths.append(os.path.join(coarse_ape,filename))
    db_set_quaternions.append(coarse_ape_qs[i])
# print(np.shape(training_set_image_paths))
# print(np.shape(training_set_quaternions))
# print(np.shape(test_set_image_paths))
# print(np.shape(test_set_quaternions))

# Appending the fine images for benchvise into db set
text = open(os.path.join(script_dir, coarse_benchvise_quaternion), 'r').read().split('\n')
coarse_benchvise_qs = [data for data in text if not data.__contains__("#")]
coarse_benchvise_qs = coarse_benchvise_qs[0:len(coarse_benchvise_qs)-1]
coarse_benchvise_qs = [data.split(" ") for data in coarse_benchvise_qs]
for i in range(np.shape(coarse_benchvise_qs)[0]):
    filename = "coarse" + str(i) + ".png"
    db_set_image_paths.append(os.path.join(coarse_benchvise,filename))
    db_set_quaternions.append(coarse_benchvise_qs[i])
# print(np.shape(training_set_image_paths))
# print(np.shape(training_set_quaternions))
# print(np.shape(test_set_image_paths))
# print(np.shape(test_set_quaternions))

# Appending the fine images for cam into db set
text = open(os.path.join(script_dir, coarse_cam_quaternion), 'r').read().split('\n')
coarse_cam_qs = [data for data in text if not data.__contains__("#")]
coarse_cam_qs = coarse_cam_qs[0:len(coarse_cam_qs)-1]
coarse_cam_qs = [data.split(" ") for data in coarse_cam_qs]
for i in range(np.shape(coarse_cam_qs)[0]):
    filename = "coarse" + str(i) + ".png"
    db_set_image_paths.append(os.path.join(coarse_cam,filename))
    db_set_quaternions.append(coarse_cam_qs[i])
# print(np.shape(training_set_image_paths))
# print(np.shape(training_set_quaternions))
# print(np.shape(test_set_image_paths))
# print(np.shape(test_set_quaternions))

# Appending the fine images for cat into db set
text = open(os.path.join(script_dir, coarse_cat_quaternion), 'r').read().split('\n')
coarse_cat_qs = [data for data in text if not data.__contains__("#")]
coarse_cat_qs = coarse_cat_qs[0:len(coarse_cat_qs)-1]
coarse_cat_qs = [data.split(" ") for data in coarse_cat_qs]
for i in range(np.shape(coarse_cat_qs)[0]):
    filename = "coarse" + str(i) + ".png"
    db_set_image_paths.append(os.path.join(coarse_cat,filename))
    db_set_quaternions.append(coarse_cat_qs[i])
# print(np.shape(training_set_image_paths))
# print(np.shape(training_set_quaternions))
# print(np.shape(test_set_image_paths))
# print(np.shape(test_set_quaternions))

# Appending the fine images for duck into db set
text = open(os.path.join(script_dir, coarse_duck_quaternion), 'r').read().split('\n')
coarse_duck_qs = [data for data in text if not data.__contains__("#")]
coarse_duck_qs = coarse_duck_qs[0:len(coarse_duck_qs)-1]
coarse_duck_qs = [data.split(" ") for data in coarse_duck_qs]
for i in range(np.shape(coarse_duck_qs)[0]):
    filename = "coarse" + str(i) + ".png"
    db_set_image_paths.append(os.path.join(coarse_duck,filename))
    db_set_quaternions.append(coarse_duck_qs[i])
# print(np.shape(training_set_image_paths))
# print(np.shape(training_set_quaternions))
# print(np.shape(test_set_image_paths))
# print(np.shape(test_set_quaternions))

# Write the db.txt files
writeFile = open("db.txt",'w')
for i in range(np.shape(db_set_image_paths)[0]):
    writeFile.write(db_set_image_paths[i])
    writeFile.write(";")
    for j in range(4):
        writeFile.write(db_set_quaternions[i][j])
        if j is not 3:
            writeFile.write(";")
        else:
            writeFile.write("\n")

