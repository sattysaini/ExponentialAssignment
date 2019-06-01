import numpy as np
from keras.layers import Input, Dense
from keras.models import Model 
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

batch_size = 1024
epochs = 10
sessions = 5

def exp():
	inputs = Input((1,))
	layer0 = Dense(512, activation = "relu", kernel_initializer = "random_uniform")(inputs)
	layer1 = Dense(1024, activation = "relu", kernel_initializer = "random_uniform")(layer0)
	layer2 = Dense(1024, activation = "relu", kernel_initializer = "random_uniform")(layer1)
	outputs = Dense(1, activation = None)(layer2)

	model = Model(inputs = [inputs], outputs = [outputs])
	adm = Adam(lr=0.001)
	model.compile(optimizer = adm, loss = 'mean_absolute_error', metrics = ['mae'])
	model.summary()
	return model



def main():
	
	model = exp()

	for sess in range(sessions):
		X = np.random.random_sample((batch_size,)) * 10 * np.pi - (5 * np.pi)
		Y = np.exp(X)
		for epoch in range(epochs):
			model.fit(X, Y, validation_split = 0.1)
			y_predicted = model.predict(X)
			fig, axs = plt.subplots(1, 2)
			axs[0].scatter(X, Y, color = "green")
			axs[1].scatter(X, y_predicted, color = "red")
			name = str(sess) + "_" + str(epoch) + ".png"
			fig.savefig(os.path.join("F:\Git\ExponentialAssignment\exp^x_graphs", name))
			plt.close()


if __name__ == '__main__':
	main()
