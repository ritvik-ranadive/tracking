import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(__file__)
lossFilePath = os.path.join(script_dir,"loss_data")
lossData = open(lossFilePath, 'r').read().split('\n')
lossData = [lossString.split("INFO:tensorflow:Loss for final step: ") for lossString in lossData if lossString.__contains__("Loss for final step")]
# print(lossData)
loss = []
for info in lossData:
    loss.append(float(info[1][0:len(info[1])-1]))

# print(loss)
plt.plot(loss)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
