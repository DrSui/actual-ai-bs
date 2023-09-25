import numpy as np
from data import get_data
import matplotlib.pyplot as plt
import time

class NeuralNetwork:
    def __init__(self, n, learn_rate, size):
        self.array = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]
        self.n = n
        self.learn_rate = learn_rate
        self.weight_input_hidden1 = np.random.uniform(-0.5, 0.5, (n, size * size * 3))
        self.weight_hidden1_hidden2 = np.random.uniform(-0.5, 0.5, (n, n))
        self.weight_hidden2_hidden3 = np.random.uniform(-0.5, 0.5, (n, n))
        self.weight_hidden3_output = np.random.uniform(-0.5, 0.5, (11, n))

        self.bias_input_hidden1 = np.zeros((n, 1))
        self.bias_hidden1_hidden2 = np.zeros((n, 1))
        self.bias_hidden2_hidden3 = np.zeros((n, 1))
        self.bias_hidden3_output = np.zeros((11, 1))
    def forward(self, x):
        # Forward propagation input -> hidden1
        hidden1_pre = self.bias_input_hidden1 + self.weight_input_hidden1 @ x
        self.hidden1 = 1 / (1 + np.exp(-hidden1_pre))

        # Forward propagation hidden1 -> hidden2
        hidden2_pre = self.bias_hidden1_hidden2 + self.weight_hidden1_hidden2 @ self.hidden1
        self.hidden2 = 1 / (1 + np.exp(-hidden2_pre))

        # Forward propagation hidden2 -> hidden3
        hidden3_pre = self.bias_hidden2_hidden3 + self.weight_hidden2_hidden3 @ self.hidden2
        self.hidden3 = 1 / (1 + np.exp(-hidden3_pre))
        
        # Forward propagation hidden2 -> output
        output_pre = self.bias_hidden3_output + self.weight_hidden3_output @ self.hidden3
        output = 1 / (1 + np.exp(-output_pre))

        return output
    def backward(self,img,output,label):
            # Backpropagation output -> hidden3 (cost function derivative)
            delta_output = output - label
            self.weight_hidden3_output -= self.learn_rate * delta_output @ self.hidden3.T
            self.bias_hidden3_output -= self.learn_rate * delta_output

            # Backpropagation hidden3 -> hidden2 (activation function derivative)
            delta_hidden3 = self.weight_hidden3_output.T @ delta_output * (self.hidden3 * (1 - self.hidden3))
            self.weight_hidden2_hidden3 -= self.learn_rate * delta_hidden3 @ self.hidden2.T
            self.bias_hidden2_hidden3 -= self.learn_rate * delta_hidden3

            # Backpropagation hidden2 -> hidden1 (activation function derivative)
            delta_hidden2 = self.weight_hidden2_hidden3 @ delta_hidden3 * (self.hidden2 * (1 - self.hidden2))
            self.weight_hidden1_hidden2 -= self.learn_rate * delta_hidden2 @ self.hidden2.T
            self.bias_hidden1_hidden2 -= self.learn_rate * delta_hidden2

            # Backpropagation hidden1 -> input (activation function derivative)
            delta_hidden1 = self.weight_hidden1_hidden2 @ delta_hidden2 * (self.hidden1 * (1 - self.hidden1))
            self.weight_input_hidden1 -= self.learn_rate * delta_hidden1 @ img.T
            self.bias_input_hidden1 -= self.learn_rate * delta_hidden1
    def check(self, images, labels):
        permuated = np.random.permutation(images.shape[0])
        nr_correct = 0
        total_loss = 0
        for i in permuated:
            img = images[i]
            label = labels[i]
            img.shape += (1,)
            label.shape += (1,)
            
            one_hot_label = np.zeros((11,))
            one_hot_label[label] = 1
            
            output = self.forward(img)
            
            nr_correct += int(np.argmax(output) == np.argmax(label))  # Compare with the original label
            loss = -np.sum(one_hot_label * np.log(output) + (1 - one_hot_label) * np.log(1 - output))
            total_loss += loss
        nr_avg = round((nr_correct / images.shape[0]) * 100, 2)
        print(f"Finally Average Of Model: {nr_avg}%")

    def train(self, epochs, images, labels):
        print("training...")
        total_loss = 0.0
        nr_correct = 0
        total_time = 0

        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(images.shape[0])
            start_time = time.time()
            for idx in shuffled_indices:
                img = images[idx]
                label = labels[idx]
                img.shape += (1,)
                label.shape += (1,)
                # One-hot encode the label
                one_hot_label = np.zeros((11,))
                one_hot_label[label] = 1
                
                output = self.forward(img)
                # Cost / Error calculation
                #error = 1 / len(output) * cp.sum((output - one_hot_label) ** 2, axis=0)
                nr_correct += int(np.argmax(output) == np.argmax(label))  # Compare with the original label
                loss = -np.sum(one_hot_label * np.log(output) + (1 - one_hot_label) * np.log(1 - output))
                total_loss += loss
            
                self.backward(img,output,label)
            # Show accuracy for this epoch
            end_time = time.time()
            delta_time = end_time - start_time
            nr_avg = round((nr_correct / images.shape[0]) * 100, 2)
            avg_loss = total_loss / images.shape[0]
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Average Loss: {np.mean(avg_loss).item():.4f}")
            print(f"Accuracy: {nr_avg}%")
            print(f"time taken: {delta_time}")
            total_time += delta_time
            nr_correct = 0
            if nr_avg >= 40:
                self.learn_rate *=.1
        print(f"total time was: {total_time} average time: {total_time/epochs}")
    def save(self):
        np.savez("neuralNetworkData.npz", 
            weight_input_hidden1=self.weight_input_hidden1, 
            weight_hidden1_hidden2 = self.weight_hidden1_hidden2,
            weight_hidden2_hidden3 = self.weight_hidden2_hidden3,
            weight_hidden3_output= self.weight_hidden3_output,
            bias_input_hidden1 = self.bias_input_hidden1,
            bias_hidden1_hidden2 = self.bias_hidden1_hidden2,
            bias_hidden2_hidden3 = self.bias_hidden2_hidden3,
            bias_hidden3_output= self.bias_hidden3_output)
    def load(self):
        print("loading...")
        data = np.load("neuralNetworkData.npz")
        
        self.weight_input_hidden1 = np.asarray(data["weight_input_hidden1"])
        self.weight_hidden1_hidden2 = np.asarray(data["weight_hidden1_hidden2"])
        self.weight_hidden2_hidden3 = np.asarray(data["weight_hidden2_hidden3"])
        self.weight_hidden3_output = np.asarray(data["weight_hidden3_output"])
        
        self.bias_input_hidden1 = np.asarray(data["bias_input_hidden1"])
        self.bias_hidden1_hidden2 = np.asarray(data["bias_hidden1_hidden2"])
        self.bias_hidden2_hidden3 = np.asarray(data["bias_hidden2_hidden3"])
        self.bias_hidden3_output = np.asarray(data["bias_hidden3_output"])
def main():
    uinput = input("load or train or both: ")
    size = 64
    n = 200
    learn_rate = 0.0005
    
    # Convert NumPy arrays to CuPy arrays
    images, labels = get_data()
    images = np.asarray(images)
    labels = np.asarray(labels)
    neural_net = NeuralNetwork(n, learn_rate, size)
    if uinput == "train":
        epochs = 100
        neural_net.train(epochs, images, labels)
        neural_net.save()

    else:
        neural_net.load()
        neural_net.check(images, labels)
        if uinput == "both":
            print("begining training...")
            neural_net.train(10, images, labels)
            neural_net.save()
    while True:
        index = int(input("Enter a number (0 - 13296): "))

        img = images[index].reshape(size, size, 3)
        #img_np = img.get()
        print("Image shape:", img.shape)
        plt.imshow(img)

        img = img.reshape(size * size * 3, 1)

        # Forward propagation input -> hidden1
        output = neural_net.forward(img)

        plt.title(f"Is it {neural_net.array[np.argmax(output).item()]} :)")
        print(output)
        print(np.argmax(output))
        plt.show()

if __name__ == "__main__":
    main()
