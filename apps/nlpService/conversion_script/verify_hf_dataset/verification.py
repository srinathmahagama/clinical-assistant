from datasets import load_from_disk

# Load your converted dataset
dataset = load_from_disk("./noongar_hf_dataset")

print("Dataset structure:", dataset)
print("Train size:", len(dataset["train"]))
print("Test size:", len(dataset["test"]))
print("First training example:", dataset["train"][0])