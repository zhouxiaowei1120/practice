import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

curve_or_mat = 'curve'

if curve_or_mat == 'curve':
    x_aixs = [0.1, 2.0, 5.0, 7.0, 10.0, 15.0, 20.0]

    ori_acc = [0.9913, 0.9913, 0.9913, 0.9913, 0.9913, 0.9913, 0.9913]

    adv_acc = [0.9934, 0.9926, 0.9907, 0.9883, 0.9845, 0.9803, 0.9794]

    attack_acc = [0.9789, 0.9644, 0.8856, 0.7822, 0.5884, 0.3120, 0.1636]

    defence_acc = [0.9640, 0.9573, 0.9198, 0.8629, 0.7527, 0.5478, 0.4153]

    fig = plt.figure()
    
    ax1 = fig.add_subplot(111)
    # ax1 = fig.add_subplot(211)
    ax1.plot(x_aixs, ori_acc, ':.', color='b', label='original model on Mnist')
    ax1.plot(x_aixs, adv_acc, 's-', color='b', label='adversarial model on Mnist')
    ax1.set_ylabel('Accuracy on Mnist test data', color='b')
    ax1.tick_params(axis='y', colors='b')
    ax1.legend(loc=3)
    
    ax2 = ax1.twinx()
    # ax2 = fig.add_subplot(212)
    ax2.plot(x_aixs, attack_acc, '--',color='red', label='original model on Adv.')
    ax2.plot(x_aixs, defence_acc, '-', color='red', label='fine-tuned model on Adv.')
    ax2.set_ylabel('Accuracy on Large Adv. datasets', color='r')    
    ax2.tick_params(axis='y', colors='r')
    ax2.legend(loc=0)
   
    plt.xlabel('epsilon')
    plt.title('Classification Accuracy Curves')
    plt.savefig('C:/Users/12882357/Desktop/acc_eps.png')
    plt.show()
elif curve_or_mat == 'mat':
    eps_list = [0.1,1.0, 2.0,3.0,4.0, 5.0, 7.0,9.0,11.0,13.0, 15.0,17.0, 20.0]
    res_dir = 'C:\\Users\\12882357\\Desktop\\research\\R_generation\\experiments\\gsn_hf\\mnist_train_test_lenet_3_1norm_ncfl50_NormL2\\res_data61_epsdiff10\\attack_rate\\'
    attack_rate = np.load(os.path.join(res_dir,'attack_rate.npy'))
    for i in range(attack_rate.shape[0]):
        matshow = plt.matshow(attack_rate[i], cmap=plt.cm.gray)
        plt.colorbar(matshow)
        plt.clim(0.0,1.0)
        plt.ylabel('source label')
        plt.xlabel('target label')
        plt.xticks(np.arange(0,10,1),('0','1','2','3','4','5','6','7','8','9'))
        plt.yticks(np.arange(0,10,1),('0','1','2','3','4','5','6','7','8','9'))
        plt.title('epsilon: '+str(eps_list[i]))
        ax = plt.gca()
        ax.xaxis.set_ticks_position('bottom')
        plt.savefig(os.path.join(res_dir, 'attack_rate_eps_'+str(eps_list[i])+ '.png'),bbox_inches='tight')
        print(attack_rate[i])
