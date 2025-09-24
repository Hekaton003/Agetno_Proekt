import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    x = [50, 100, 150, 200, 250, 300]
    y1 = [4.28,3.93,1.41,4.4,2.42,0.84]
    y2 = [30.74,32.83,32.14,32.53,32.68,32.28]
    plt.figure(figsize=(10, 10))
    plt.title('default TD3 algorithm vs optimized TD3 algorithm', fontsize=16)
    plt.xlabel('Iterations')
    plt.ylabel('Average reward')
    plt.grid(True, linestyle="--", alpha=0.6)
    sns.lineplot(x=x, y=y1, color='blue', label='TD3  algorithm')
    sns.lineplot(x=x, y=y2, color='red', label='TD3  optimized  algorithm')
    plt.legend(fontsize=12)
    plt.savefig("Visualizations/img16.jpg")
    plt.show()
