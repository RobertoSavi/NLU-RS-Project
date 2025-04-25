import matplotlib.pyplot as plt
# -------------------- Plot PPL results --------------------
plt.figure(figsize=(10, 5))
plt.plot(range(len(best_ppls)), best_ppls, marker='o', linestyle='-', label='Best PPL per Model')
plt.xlabel('Model index')
plt.ylabel('Perplexity (PPL)')
plt.title('Best PPL for each Model-Optimizer configuration')
plt.legend()
plt.savefig('images/models_best_lr.jpg')  # Save the plot as an image
plt.show()
