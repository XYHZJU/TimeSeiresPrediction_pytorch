import numpy as np
import random
from matplotlib import pyplot as plt

def random_walk( 
    data, start_value=0, threshold=0.5,  
    step_size=1,noise_size = 1.2, min_value=-np.inf, max_value=np.inf 
    ): 
    previous_value = start_value
    trend_value = start_value
    x,y = data.shape
    trend = np.zeros((x,y))
    for dim in range(0,x): 
        for index in range(0,y): 
            if previous_value < min_value: 
                previous_value = min_value 
            if previous_value > max_value: 
                previous_value = max_value 
            probability = random.random()
            randomsize = random.random() 
            if probability >= threshold: 
                data[dim,index] = data[dim,index] + noise_size *randomsize*10
                trend[dim,index] = trend_value + step_size *randomsize*10
            else: 
                data[dim,index] = data[dim,index] - noise_size *randomsize*10
                trend[dim,index] = trend_value - step_size *randomsize*10
            previous_value = data[dim,index]
            trend_value = trend[dim,index]
    return trend+data

def generate_data(dimension,length):
    alpha = []
    alpha_amp = []
    beta = []
    beta_amp = []
    gama = []
    for i in range(0,dimension):
        scale = random.randint(30,50)
        alpha.append(random.randint(10,20)/100)
        alpha_amp.append(random.random()*scale)
        beta.append(random.randint(10,20)/100)
        beta_amp.append(random.random()*scale)
        gama.append(random.randint(200,400))
    series = range(0,length)
    series = np.array(series)
    data = np.zeros((dimension,length))
    for i in range(0,dimension):
        
        temp = alpha_amp[i] * np.sin(alpha[i]*series) + beta_amp[i] * np.cos(beta[i]*series) + gama[i]*np.ones(length)
        data[i:] = np.array(temp)
    # data = data.T
    return data

def draw_graph(data):
    # print(data.shape)
    x,y = data.shape
    plt.cla()
    for i in range(1, x+1):
        plt.subplot(x, 1, i)
        length = range(0,y)
        plt.plot(length,data[i-1,:])
    plt.show()

data = generate_data(5,500)
# data = np.zeros((5,500))
draw_graph(random_walk(data))
# draw_graph(data)




