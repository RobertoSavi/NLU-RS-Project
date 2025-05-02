import os
import re
import matplotlib.pyplot as plt

folder = "results"
ppl_data = {}

# Loop through all files
for filename in os.listdir(folder):
    if filename.startswith("mod-RNN") or filename.startswith("overall") or not os.path.isfile(os.path.join(folder, filename)):
        continue

    filepath = os.path.join(folder, filename)

    with open(filepath, "r") as f:
        content = f.read()

    # Extract PPL values using regex
    ppl_values = [float(match) for match in re.findall(r"PPL:\s*([\d.]+)", content)]

    # Save PPL values under the filename (without extension if desired)
    ppl_data[filename] = ppl_values

# Plotting
plt.figure(figsize=(10, 7))

for fname, values in ppl_data.items():
    plt.plot(range(1, len(values) + 1), values, label=fname)

plt.xlabel("Epoch")
plt.ylabel("PPL")
plt.title("PPL per Epoch")
plt.yscale("log")
plt.legend()

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.legend(fontsize='small')
plt.tight_layout()  # Keeps plot compact despite legend
plt.ylim(0, 1000)  # Adatta in base al tuo range utile
plt.grid(True)
plt.tight_layout()
plt.show()

""" # -------------------- Plot PPL results --------------------
plt.figure(figsize=(10, 5))
plt.plot(range(len(best_ppls)), best_ppls, marker='o', linestyle='-', label='Best PPL per Model')
plt.xlabel('Model index')
plt.ylabel('Perplexity (PPL)')
plt.title('Best PPL for each Model-Optimizer configuration')
plt.legend()
plt.savefig('images/models_best_lr.jpg')  # Save the plot as an image
plt.show()
 """