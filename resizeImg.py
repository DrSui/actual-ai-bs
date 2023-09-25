from PIL import Image
import numpy as np
import os
import random

array = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]
save_path = "./data/test/"

# Create an empty list to store processed images and labels
data_list = []
labels_list = []

# Function to resize an image
def Resize(img, width, height):
    resized_img = img.resize((width, height))
    return resized_img

dirr = "data/training/"
dirrr = ["training", "validation"]
for d in dirrr:
    for images in array:
        print(images)
        idk = 0
        for x in os.listdir(f"data/{d}/{images}/"):
            idk +=1
            print(x)
            img = Image.open(dirr + images + "/" + x)
            image_array = np.array(img)
            blue = image_array[:, :, 0]
            green = image_array[:, :, 1]
            red = image_array[:, :, 2]
            #gray_scale_value = blue * 0.114 + green * 0.587 + red * 0.299
            #image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2] = gray_scale_value, gray_scale_value, gray_scale_value
            img = Image.fromarray(image_array)
            img_resized = Resize(img, 64, 64)
            print(np.asarray(img_resized).shape)
            #img_resized.save(save_path + images + "/" + x)

            # Append the processed image and label
            temp = [0] * 11
            temp[array.index(images)] = 1
            data_list.append(np.asarray(img_resized))
            labels_list.append(temp)
            if idk >=25:
                break

# Convert the lists to NumPy arrays
data_array = np.array(data_list)
labels_array = np.array(labels_list)

# Save both data and labels in a dictionary and then save it to .npz file
np.savez("dataSample.npz", data=data_array, labels=labels_array)
